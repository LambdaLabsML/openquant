import logging

import tqdm
import torch
import torch.nn.functional as F

from .subgraph import QuantTarget


LOGGER = logging.getLogger(__name__)


class QuantConfig:
    def __init__(self, dtype: torch.dtype):
        if dtype == torch.float8_e5m2:
            self.mantissa_bits = 2
        elif dtype == torch.float8_e4m3fn:
            self.mantissa_bits = 3
        else:
            raise NotImplementedError(dtype)

        self.dtype = dtype
        finfo = torch.finfo(dtype)
        self.min_value = finfo.min
        self.max_value = finfo.max

    def __repr__(self):
        return f"QuantConfig(dtype={self.dtype})"

    def quantize_tensor(self, x: torch.Tensor, scale: torch.Tensor):
        x = torch.clamp(x / scale, min=self.min_value, max=self.max_value)
        x = torch.round(x * (2**self.mantissa_bits)) / (2**self.mantissa_bits)
        return x

    def dequantize_tensor(self, x: torch.Tensor, scale: torch.Tensor):
        return x * scale

    def compute_qparams(self, x: torch.Tensor) -> torch.Tensor:
        return (torch.quantile(x.abs(), 0.99) / self.max_value).clamp(min=1e-6)


@torch.inference_mode()
def quantize(qcfg: QuantConfig, target: QuantTarget, inputs):
    modules_to_scale: list[torch.nn.Linear] = target.ops

    inp_dim = modules_to_scale[0].in_features
    assert inp_dim % qcfg.group_size == 0
    assert all(m.in_features == inp_dim for m in modules_to_scale)

    xs: list[torch.Tensor] = []
    for args, kwargs in inputs:
        assert len(args) == 1, len(args)
        assert isinstance(args[0], torch.Tensor)
        assert args[0].shape[-1] == inp_dim, args[0].shape
        xs.append(args[0].reshape(-1, inp_dim))

    execution_device: torch.device = xs[0].device

    total_batch_size = sum(x.shape[0] for x in xs)
    assert total_batch_size > 0

    input_scale = qcfg.compute_qparams(torch.cat(xs)).cpu()
    input_scales = [input_scale for _ in range(len(modules_to_scale))]

    weight_scales = []
    for m in modules_to_scale:
        w = m.weight.to(execution_device)
        scale = qcfg.compute_qparams(w)
        m.weight.data = qcfg.quantize_tensor(w, scale).to(m.weight.device)
        weight_scales.append(scale.cpu())

    return weight_scales, input_scales


def transformers_quant_config(qcfg: QuantConfig) -> dict:
    return {
        "quant_method": "fp8",
        "is_checkpoint_fp8_serialized": True,
        "activation_scheme": "static",
    }


def pack(
    qcfg: QuantConfig,
    model: torch.nn.Module,
    targets: list[tuple[torch.nn.Linear, torch.Tensor, torch.Tensor]],
    pack_dtype=torch.int32,
):
    pass


class QuantizedLinear(torch.nn.Module):
    def __init__(
        self,
        weight: torch.Tensor,
        weight_scale: torch.Tensor,
        input_scale: torch.Tensor,
        bias: torch.Tensor = None,
    ):
        super().__init__()
        self.register_buffer("weight", weight)
        self.register_buffer("weight_scale", weight_scale)
        self.register_buffer("input_scale", input_scale)
        if bias is not None:
            self.register_buffer("bias", bias)
        else:
            self.bias = None

    def forward(self, *args, **kwargs):
        raise NotImplementedError("This class is only used for serialization")
