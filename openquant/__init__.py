import weakref
import logging

import tqdm
import torch
import torch.nn.functional as F


LOGGER = logging.getLogger(__name__)


def get_attn_implementation():
    try:
        import flash_attn

        LOGGER.info(f"Using flash attention")
        return "flash_attention_2"
    except ImportError:
        LOGGER.info(f"`import flash_attn` not found, pip install to use")
        return None


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


# def extract_tensors(obj) -> list[torch.Tensor]:
#     tensors = []
#     if obj is None:
#         pass
#     elif isinstance(obj, torch.Tensor):
#         tensors.append(obj)
#     elif isinstance(obj, tuple):
#         for o in obj:
#             tensors.extend(extract_tensors(o))
#     elif isinstance(obj, dict):
#         for k, v in obj.items():
#             tensors.extend(extract_tensors(v))
#     else:
#         pass
#     return tensors


# class GraphTracer:
#     graph: torch.nn.Module

#     parent: dict[str, list[str]]
#     children: dict[str, list[str]]

#     output_to_module_name: dict[weakref.ReferenceType, str]

#     @classmethod
#     def init_hooks(cls, graph: torch.nn.Module):
#         for name, module in graph.named_modules():
#             if len(name) == 0:
#                 continue

#             def make_pre_forward_hook(n):
#                 def hook(module, args):
#                     parents = []
#                     for tensor in extract_tensors(args):
#                         parent = GraphTracer.output_to_module_name.get(
#                             weakref.ref(tensor)
#                         )
#                         if parent is not None:
#                             parents.append(parent)

#                     if len(parents) > 1:
#                         raise NotImplementedError()
#                     if len(parents) != 0:
#                         GraphTracer.parent[n] = parents[0]
#                         if parents[0] not in GraphTracer.children:
#                             GraphTracer.children[parents[0]] = []
#                         GraphTracer.children[parents[0]].append(n)

#                 return hook

#             def make_post_forward_hook(n):
#                 def hook(module, args, output):
#                     for tensor in extract_tensors(output):
#                         GraphTracer.output_to_module_name[weakref.ref(tensor)] = n

#                 return hook

#             module.register_forward_pre_hook(make_pre_forward_hook(name))
#             module.register_forward_hook(make_post_forward_hook(name))

#         cls.parent = {}
#         cls.children = {}
#         cls.graph = graph
#         cls.output_to_module_name = {}

#     @classmethod
#     def clear(cls):
#         cls.output_to_module_name.clear()

#     @classmethod
#     def get_parent(cls, x: torch.Tensor) -> torch.nn.Module:
#         name = cls.output_to_module_name[weakref.ref(x)]
#         return name, cls.graph.get_submodule(name)


# class LinearQuantizer:
#     def __init__(
#         self,
#         qcfg: QuantConfig,
#         name: str,
#         module: torch.nn.Linear,
#         execution_device: torch.device,
#         storage_device: torch.device,
#     ):
#         super().__init__()
#         assert isinstance(module, torch.nn.Linear)
#         self.qcfg = qcfg
#         self.name = name
#         self.module = module
#         self.execution_device = execution_device
#         self.storage_device = storage_device
#         self.out_dim, self.inp_dim = self.module.weight.shape

#     def pre_forward_hook(self, *args, **kwargs):
#         pass

#     def post_forward_hook(self, *args, **kwargs):
#         pass

#     def quantize_module(self):
#         raise NotImplementedError()

#     @classmethod
#     def get_transformers_quant_config(cls, qcfg: QuantConfig) -> dict:
#         raise NotImplementedError()


class AWQTarget:
    def __init__(self, *, inverse_scale: torch.nn.Module, scale: list[torch.nn.Module]):
        self.inverse_scale = inverse_scale
        self.scale = scale

    def names(self, root: torch.nn.Module):
        tmp = [""] * len(self.scale)
        for name, haystack in root.named_modules():
            for i in range(self.scale):
                if haystack == self.scale[i]:
                    tmp[i] = name
                    break
        return tmp


class AWQ:
    """
    AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration

    https://arxiv.org/abs/2306.00978
    """

    def __init__(
        self,
        qcfg: QuantConfig,
        target: AWQTarget,
        execution_device: torch.device,
        storage_device: torch.device = torch.device("cpu"),
        search_grid_size: int = 20,
    ):
        self.target = target
        module_to_inverse_scale: torch.nn.Module = target.inverse_scale
        modules_to_scale: list[torch.nn.Linear] = target.scale

        if modules_to_scale[0].bias is not None:
            assert all(m.bias is not None for m in modules_to_scale)
        else:
            assert all(m.bias is None for m in modules_to_scale)

        self.qcfg = qcfg
        self.module_to_inverse_scale = module_to_inverse_scale
        self.modules_to_scale = modules_to_scale
        self.execution_device = execution_device
        self.storage_device = storage_device
        self.search_grid_size = search_grid_size
        self.xs: list[torch.Tensor] = []

    def pre_forward_hook(self, module, args):
        assert module in self.modules_to_scale
        assert isinstance(args, tuple) and len(args) == 1
        x = args[0]
        assert isinstance(x, torch.Tensor)
        self.xs.append(x.to(self.storage_device).reshape(-1, self.inp_dim))
        raise ForwardPassEarlyStop()

    @torch.inference_mode()
    def quantize_module(self):
        total_batch_size = sum(x.shape[0] for x in self.xs)
        assert total_batch_size > 0

        _, inp_dim = self.modules_to_scale[0].shape
        assert inp_dim % self.qcfg.group_size == 0
        assert all(m.shape[1] == inp_dim for m in self.modules_to_scale)

        w = torch.cat([m.weight for m in self.modules_to_scale]).to(
            self.execution_device
        )
        b = None
        if self.modules_to_scale[0][1].bias is not None:
            b = torch.cat([m.bias for m in self.modules_to_scale]).to(
                self.execution_device
            )

        # TODO filter out padded!!!

        s_x: torch.Tensor = 0
        for x in self.xs:
            # NOTE: x has shape [batch, inp_dim]
            x = x.to(self.execution_device)
            # NOTE: this is the group-wise functionality
            g_x = x.reshape(x.shape[0], -1, self.qcfg.group_size)
            s_x += g_x.abs().sum(0) / total_batch_size
        s_x = s_x.reshape(inp_dim)

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

            s_m = s.repeat(len(self.modules_to_scale))

            w_s = w * s_m
            w_q = self.qcfg.quantize_tensor(w_s, *self.qcfg.compute_qparams(w_s))

            loss = 0
            for x in self.xs:
                x = x.to(self.execution_device)
                y = F.linear(x, w, bias=b)
                y_q = F.linear(x / s_m, w_q, bias=b)
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
        for module in self.modules_to_scale:
            module.weight.mul(best_s.view(1, -1))

        # Apply scale to parent op
        parent = self.module_to_inverse_scale
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
        apply_clip = not any(
            p in self.name for p in ["q_", "k_", "query", "key", "Wqkv"]
        )
        if apply_clip:
            for x in self.xs:
                x.div_(best_s)

            raise NotImplementedError()

        raise NotImplementedError()

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
