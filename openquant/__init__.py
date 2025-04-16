import weakref

import tqdm
import torch
import torch.nn.functional as F


class ForwardPassEarlyStop(Exception):
    pass


class QuantConfig:
    def __init__(
        self,
        num_bits: int = 4,
        symmetric: bool = True,
        group_size: int = 128,
    ):
        assert group_size > 0
        assert num_bits > 0

        self.num_bits = num_bits
        self.group_size = group_size
        self.symmetric = symmetric

        if self.symmetric:
            self.min_int = 0
            self.max_int = 2**self.num_bits - 1
        else:
            self.min_int = -(2 ** (self.num_bits - 1))
            self.max_int = 2 ** (self.num_bits - 1) - 1

    @torch.inference_mode()
    def quantize_tensor(self, x: torch.Tensor, scale: torch.Tensor, zero: torch.Tensor):
        x = torch.clamp(torch.round(x / scale) + zero, self.min_int, self.max_int)
        return (x - zero) * scale

    @torch.inference_mode()
    def dequantize_tensor(
        self, x: torch.Tensor, scale: torch.Tensor, zero: torch.Tensor
    ):
        return (x - zero) * scale

    @torch.inference_mode()
    def compute_qparams(self, x: torch.Tensor):
        shape = x.shape

        x = x.reshape(-1, self.group_size)

        if self.symmetric:
            min_val = x.amin(dim=1, keepdim=True)
            max_val = x.amax(dim=1, keepdim=True)
            scale = (max_val - min_val) / self.max_int
            zero = (-torch.round(min_val / scale)).clamp(self.min_int, self.max_int)
        else:
            max_val = x.abs().amax(dim=1, keepdim=True)
            scale = max_val / self.max_int
            zero = torch.zeros_like(max_val)

        return scale.reshape(shape), zero.reshape(shape)


class GraphTracer(torch.nn.Module):
    graph: torch.nn.Module
    weakref_to_module_name: dict[weakref.ReferenceType, str]

    @classmethod
    def set_graph(cls, graph: torch.nn.Module):
        cls.graph = graph

    @classmethod
    def clear(cls):
        cls.weakref_to_module_name.clear()

    @classmethod
    def get_parent(cls, x: torch.Tensor) -> torch.nn.Module:
        name = cls.weakref_to_module_name[weakref.ref(x)]
        return getattr(cls.graph, name)

    def __init__(self, name: str, module: torch.nn.Module):
        super().__init__()
        self.name = name
        self.module = module

    def forward(self, *args, **kwargs):
        output = self.module(*args, **kwargs)
        self.weakref_to_module_name[weakref.ref(output)] = self.name
        return output


class LinearQuantizer(torch.nn.Module):
    def __init__(
        self,
        qcfg: QuantConfig,
        name: str,
        module: torch.nn.Linear,
        execution_device: torch.device,
        storage_device: torch.device,
    ):
        super().__init__()
        assert isinstance(module, torch.nn.Linear)
        self.qcfg = qcfg
        self.name = name
        self.module = module
        self.execution_device = execution_device
        self.storage_device = storage_device
        self.out_dim, self.inp_dim = self.module.weight.shape

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def quantize_module(self):
        raise NotImplementedError()

    @classmethod
    def get_transformers_quant_config(cls, qcfg: QuantConfig) -> dict:
        raise NotImplementedError()



class AWQ(LinearQuantizer):
    """
    AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration

    https://arxiv.org/abs/2306.00978
    """

    def __init__(
        self,
        qcfg: QuantConfig,
        name: str,
        module: torch.nn.Linear,
        execution_device: torch.device,
        cache_device: torch.device,
        search_grid_size: int = 20,
    ):
        super().__init__(qcfg, name, module, execution_device, cache_device)
        self.search_grid_size = search_grid_size
        self.xs: list[torch.Tensor] = []

    @torch.inference_mode()
    def forward(self, x: torch.Tensor):
        self.xs.append(x.to(self.storage_device))
        raise ForwardPassEarlyStop()

    @torch.inference_mode()
    def quantize_module(self):
        assert self.inp_dim % self.qcfg.group_size == 0

        total_batch_size = sum(x.shape[0] for x in self.xs)

        w = self.module.weight.to(self.execution_device)
        b = None
        if self.module.bias is not None:
            b = self.module.bias.to(self.execution_device)

        s_x: torch.Tensor = 0
        for x in self.xs:
            # NOTE: x has shape [batch, inp_dim]
            x = x.to(self.execution_device)
            # NOTE: this is the group-wise functionality
            g_x = x.reshape(x.shape[0], -1, self.qcfg.group_size)
            s_x += g_x.abs().sum(0) / total_batch_size
        s_x = s_x.reshape(self.inp_dim)

        # Find best scale
        best_loss = float("inf")
        best_s = None
        best_alpha = None
        for alpha in tqdm.tqdm(
            torch.linspace(0, 1, steps=self.search_grid_size + 1),
            leave=False,
            desc="Searching scales",
        ):
            s = s_x.pow(alpha).clamp(min=1e-8)
            s[torch.isinf(s) | torch.isnan(s)] = 1

            w_s = w * s
            w_q = self.qcfg.quantize_tensor(w_s, *self.qcfg.compute_qparams(w_s))

            loss = 0
            for x in self.xs:
                x = x.to(self.execution_device)
                y = F.linear(x, w, bias=b)
                y_q = F.linear(x / s, w_q, bias=b)
                loss += (y - y_q).float().square().sum() / (
                    total_batch_size * self.out_dim
                )

            if loss < best_loss:
                best_loss = loss
                best_s = s.to(self.storage_device)
                best_alpha = alpha

        # Apply scale to module
        self.module.weight.mul_(best_s.view(1, -1))

        # Apply scale to parent op
        parent = GraphTracer.get_parent(self.xs[0])
        parent_op_name = parent.__class__.__name__.lower()
        if isinstance(parent, torch.nn.Linear):
            parent.weight.div_(best_s.view(-1, 1))
            if parent.bias is not None:
                parent.bias.div_(best_s)

        elif "norm" in parent_op_name:
            assert hasattr(parent, "weight")
            if "gemma" in parent_op_name:
                parent.weight += 1
                parent.weight.div_(best_s)
                parent.weight -= 1
            else:
                parent.weight.div_(best_s)
            if hasattr(parent, "bias") and parent.bias is not None:
                parent.bias.div_(best_s)

        else:
            raise NotImplementedError(f"Can't rescale previous op {parent}")

        # # Search for best clip
        # apply_clip = not any(
        #     p in self.name for p in ["q_", "k_", "query", "key", "Wqkv"]
        # )
        # for x in self.xs:
        #     x.div_(best_s)

        # # TODO apply clip to module

        # raise NotImplementedError()

    @classmethod
    def get_transformers_quant_config(cls, qcfg: QuantConfig) -> dict:
        return {
            "quant_method": "awq",
            "zero_point": qcfg.symmetric,
            "group_size": qcfg.group_size,
            "bits": qcfg.num_bits,
            "version": "gemm",
            "modules_to_not_convert": None,
        }

