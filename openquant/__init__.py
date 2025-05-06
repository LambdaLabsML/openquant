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
        LOGGER.warning(
            f"`import flash_attn` not found, run `pip install flash-attn --no-build-isolation`"
        )
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
    def __init__(
        self, *, inverse_scale: torch.nn.Module, scales: list[torch.nn.Module]
    ):
        self.inverse_scale = inverse_scale
        self.scales = scales

    def names(self, root: torch.nn.Module):
        tmp = [""] * len(self.scales)
        for name, haystack in root.named_modules():
            for i in range(len(self.scales)):
                if haystack == self.scales[i]:
                    tmp[i] = name
                    break
        return tmp


class InputCatcher:
    def __init__(self, module: torch.nn.Module):
        if not isinstance(module, list):
            module = [module]
        self.modules = module
        self.inputs = []
        self.handles = [
            m.register_forward_pre_hook(self.pre_forward_hook, with_kwargs=True)
            for m in self.modules
        ]

    def pre_forward_hook(self, module, args, kwargs):
        assert module in self.modules
        self.inputs.append([args, kwargs])
        raise ForwardPassEarlyStop()

    def remove_handle_and_get(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        return self.inputs


@torch.inference_mode()
def awq(
    qcfg: QuantConfig,
    modules_to_scale: list[torch.nn.Linear],
    module_to_inverse_scale: torch.nn.Module,
    execution_device: torch.device,
    inputs,
    storage_device: torch.device = torch.device("cpu"),
    search_grid_size: int = 20,
):
    """
    AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration

    https://arxiv.org/abs/2306.00978
    """
    inp_dim = modules_to_scale[0].in_features
    assert inp_dim % qcfg.group_size == 0
    assert all(m.in_features == inp_dim for m in modules_to_scale)

    xs = []
    for args, kwargs in inputs:
        assert len(args) == 1, len(args)
        assert isinstance(args[0], torch.Tensor)
        assert args[0].shape[-1] == inp_dim, args[0].shape
        xs.append(args[0].reshape(-1, inp_dim).to(execution_device))

    total_batch_size = sum(x.shape[0] for x in xs)
    assert total_batch_size > 0

    w = torch.cat([m.weight for m in modules_to_scale]).to(execution_device)
    b = None
    if modules_to_scale[0].bias is not None:
        b = torch.cat([m.bias for m in modules_to_scale]).to(execution_device)

    # NOTE: input magnitude calculation (per input channel i.e. per `inp_dim`)
    s_x: torch.Tensor = 0
    # TODO filter out padded!!!
    for x in xs:
        # NOTE: x has shape [batch, inp_dim]
        # NOTE: this reshape is the group-wise functionality
        g_x = x.reshape(x.shape[0], -1, qcfg.group_size)
        s_x += g_x.abs().sum(0) / total_batch_size
    s_x = s_x.reshape(inp_dim)

    # Find best scale
    best_loss = float("inf")
    best_s = None
    best_alpha = None
    pbar = tqdm.tqdm(total=search_grid_size + 1, leave=False, desc="Searching scales")
    for alpha in torch.linspace(0, 1, steps=search_grid_size + 1):
        s = s_x.pow(alpha).clamp(min=1e-8)
        s[torch.isinf(s) | torch.isnan(s)] = 1

        w_s = w * s
        w_q = qcfg.quantize_tensor(w_s, *qcfg.compute_qparams(w_s))

        loss = 0
        for x in xs:
            y = F.linear(x, w, bias=b)
            y_q = F.linear(x / s, w_q, bias=b)
            loss += (y - y_q).float().square().sum()
        loss /= total_batch_size * w.shape[0]

        if loss < best_loss:
            best_loss = loss
            best_s = s
            best_alpha = alpha

        pbar.set_description(
            f"Searching scales: best_loss={best_loss} @ alpha={best_alpha:.3f}",
            refresh=False,
        )
        pbar.update()

    # Apply scale to module
    for module in modules_to_scale:
        w = module.weight.data.to(execution_device)
        w_s = w * best_s
        w_q = qcfg.quantize_tensor(w_s, *qcfg.compute_qparams(w_s))
        module.weight.data = w_q.to(storage_device)

    # Apply scale to parent op
    parent = module_to_inverse_scale
    parent_op_name = parent.__class__.__name__.lower()
    if isinstance(parent, torch.nn.Linear):
        assert parent.out_features == inp_dim, (
            inp_dim,
            parent.in_features,
            parent.out_features,
        )
        assert parent.weight.shape[0] == inp_dim, (inp_dim, parent.weight.shape)
        parent.weight.div_(best_s.view(inp_dim, 1))
        if parent.bias is not None:
            assert parent.bias.shape[0] == inp_dim, (inp_dim, parent.bias.shape)
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


def awq_transformers_quant_config(qcfg: QuantConfig) -> dict:
    return {
        "quant_method": "awq",
        "zero_point": qcfg.zero_point,
        "group_size": qcfg.group_size,
        "bits": qcfg.num_bits,
        "version": "gemm",
        "modules_to_not_convert": None,
    }


def pack(qcfg: QuantConfig, model):
    raise NotImplementedError()
