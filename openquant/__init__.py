import weakref
import logging

import tqdm
import torch
import torch.nn.functional as F


LOGGER = logging.getLogger(__name__)


class ForwardPassEarlyStop(Exception):
    pass


class QuantConfig:
    def __init__(
        self,
        num_bits: int = 4,
        zero_point: bool = True,
        group_size: int = 128,
    ):
        assert group_size > 0
        assert num_bits > 0

        self.num_bits = num_bits
        self.group_size = group_size
        self.zero_point = zero_point

        if self.zero_point:
            self.min_int = 0
            self.max_int = 2**self.num_bits - 1
        else:
            self.min_int = -(2 ** (self.num_bits - 1))
            self.max_int = 2 ** (self.num_bits - 1) - 1

    def __repr__(self):
        return f"QuantConfig(num_bits={self.num_bits}, zero_point={self.zero_point}, group_size={self.group_size})"

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

        if self.zero_point:
            min_val = x.amin(dim=1, keepdim=True).broadcast_to(-1, self.group_size)
            max_val = x.amax(dim=1, keepdim=True).broadcast_to(-1, self.group_size)
            scale = (max_val - min_val) / self.max_int
            zero = (-torch.round(min_val / scale)).clamp(self.min_int, self.max_int)
        else:
            max_val = (
                x.abs().amax(dim=1, keepdim=True).broadcast_to(-1, self.group_size)
            )
            scale = max_val / self.max_int
            zero = torch.zeros_like(max_val)

        return scale.reshape(shape), zero.reshape(shape)


class GraphTracer:
    graph: torch.nn.Module
    weakref_to_module_name: dict[weakref.ReferenceType, str]

    @classmethod
    def init_hooks(cls, graph: torch.nn.Module):
        for name, module in graph.named_modules():
            if len(name) == 0:
                continue

            def make_hook(n):
                def hook(module, args, output):
                    if isinstance(output, tuple):
                        for o in output:
                            if o is None:
                                continue
                            assert isinstance(o, torch.Tensor), type(o)
                            GraphTracer.weakref_to_module_name[weakref.ref(o)] = n
                    else:
                        assert isinstance(output, torch.Tensor), type(output)
                        GraphTracer.weakref_to_module_name[weakref.ref(output)] = n

                return hook

            module.register_forward_hook(make_hook(name))

        cls.graph = graph
        cls.weakref_to_module_name = {}

    @classmethod
    def clear(cls):
        cls.weakref_to_module_name.clear()

    @classmethod
    def get_parent(cls, x: torch.Tensor) -> torch.nn.Module:
        name = cls.weakref_to_module_name[weakref.ref(x)]
        return name, cls.graph.get_submodule(name)


class LinearQuantizer:
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

    def pre_forward_hook(self, *args, **kwargs):
        pass

    def post_forward_hook(self, *args, **kwargs):
        pass

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
        storage_device: torch.device,
        search_grid_size: int = 20,
    ):
        super().__init__(qcfg, name, module, execution_device, storage_device)
        self.search_grid_size = search_grid_size
        self.xs: list[torch.Tensor] = []
        self.parent = None

    def pre_forward_hook(self, module, args):
        assert self.module == module
        assert isinstance(args, tuple) and len(args) == 1
        x = args[0]
        assert isinstance(x, torch.Tensor)
        if self.parent is None:
            parent_name, self.parent = GraphTracer.get_parent(x)
            LOGGER.debug(f"Found parent layer {parent_name} {self.parent}")
        self.xs.append(x.to(self.storage_device).reshape(-1, self.inp_dim))
        raise ForwardPassEarlyStop()

    @torch.inference_mode()
    def quantize_module(self):
        assert self.inp_dim % self.qcfg.group_size == 0

        total_batch_size = sum(x.shape[0] for x in self.xs)
        assert total_batch_size > 0

        w = self.module.weight.to(self.execution_device)
        b = None
        if self.module.bias is not None:
            b = self.module.bias.to(self.execution_device)

        # TODO filter out padded!!!

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
        pbar = tqdm.tqdm(
            total=self.search_grid_size + 1, leave=False, desc="Searching scales"
        )
        for alpha in torch.linspace(0, 1, steps=self.search_grid_size + 1):
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

            pbar.set_description(
                f"Searching scales: loss@{best_loss:.5f} alpha@{best_alpha:.3f}",
                refresh=False,
            )
            pbar.update()

        # Apply scale to module
        self.module.weight.mul_(best_s.view(1, -1))

        # Apply scale to parent op
        parent_op_name = self.parent.__class__.__name__.lower()
        if isinstance(self.parent, torch.nn.Linear):
            self.parent.weight.div_(best_s.view(-1, 1))
            if self.parent.bias is not None:
                self.parent.bias.div_(best_s)

        elif "norm" in parent_op_name:
            assert hasattr(self.parent, "weight")
            if "gemma" in parent_op_name:
                self.parent.weight += 1
                self.parent.weight.div_(best_s)
                self.parent.weight -= 1
            else:
                self.parent.weight.div_(best_s)
            if hasattr(self.parent, "bias") and self.parent.bias is not None:
                self.parent.bias.div_(best_s)

        else:
            raise NotImplementedError(f"Can't rescale previous op {self.parent}")

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
            "zero_point": qcfg.zero_point,
            "group_size": qcfg.group_size,
            "bits": qcfg.num_bits,
            "version": "gemm",
            "modules_to_not_convert": None,
        }
