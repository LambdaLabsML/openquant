import logging
from typing import Optional

import torch

from transformers.models.qwen3_moe.modeling_qwen3_moe import (
    Qwen3MoeSparseMoeBlock,
    Qwen3MoeMLP,
)
from transformers.models.llama4.modeling_llama4 import Llama4TextExperts

from .subgraph import QuantTarget
from .utils import set_submodule


LOGGER = logging.getLogger(__name__)


class QuantConfig:
    def __init__(
        self,
        dtype: torch.dtype = torch.float8_e4m3fn,
        weight_block_size: Optional[tuple[int, int]] = None,
    ):
        if dtype == torch.float8_e5m2:
            self.mantissa_bits = 2
        elif dtype == torch.float8_e4m3fn:
            self.mantissa_bits = 3
        else:
            raise NotImplementedError(dtype)

        if weight_block_size is not None:
            assert len(weight_block_size) == 2 and all(
                isinstance(w, int) for w in weight_block_size
            )

        self.dtype = dtype
        finfo = torch.finfo(dtype)
        self.min_value = finfo.min
        self.max_value = finfo.max
        self.weight_block_size = weight_block_size

    def __repr__(self):
        return f"QuantConfig(dtype={self.dtype}, weight_block_size={self.weight_block_size})"

    def quantize_tensor(self, x: torch.Tensor, scale: torch.Tensor):
        assert x.ndim >= 2
        *shape, N, K = x.shape
        if self.weight_block_size is not None:
            n, k = self.weight_block_size
            assert N % n == 0 and K % k == 0
            x = x.reshape(*shape, N // n, n, K // k, k)
            assert scale.shape == torch.Size([*shape, N // n, K // k])
            scale = scale.reshape(*shape, N // n, 1, K // k, 1)
        else:
            assert scale.shape == torch.Size([*shape])
            scale = scale.reshape(*shape, 1, 1)
        x = torch.clamp(x / scale, min=self.min_value, max=self.max_value)
        x = torch.round(x * (2**self.mantissa_bits)) / (2**self.mantissa_bits)
        return x.reshape(*shape, N, K)

    def dequantize_tensor(self, x: torch.Tensor, scale: torch.Tensor):
        return x * scale

    def compute_qparams(self, x: torch.Tensor) -> torch.Tensor:
        assert x.ndim >= 2 and x.is_contiguous()
        *shape, N, K = x.shape
        if self.weight_block_size is not None:
            n, k = self.weight_block_size
            assert N % n == 0 and K % k == 0
            x = x.reshape(*shape, N // n, n, K // k, k)
            dims_to_reduce = [x.ndim - 3, x.ndim - 1]
        else:
            dims_to_reduce = [x.ndim - 2, x.ndim - 1]
        return (x.abs().amax(dim=dims_to_reduce) / self.max_value).clamp(min=1e-6)


@torch.inference_mode()
def pack(
    qcfg: QuantConfig,
    model: torch.nn.Module,
    targets: list[QuantTarget],
    device: torch.device,
):
    model_original_num_bytes = sum(
        p.numel() * p.dtype.itemsize for p in model.parameters()
    )

    for target in targets:
        if all(isinstance(m, torch.nn.Linear) for m in target.ops):
            for module in target.ops:
                w = module.weight.to(device)
                scale = qcfg.compute_qparams(w)
                q = qcfg.quantize_tensor(w, scale).to(qcfg.dtype)
                packed = PackedLinearLayer(q.cpu(), scale.cpu())
                set_submodule(model, module, packed)
        elif isinstance(target.ops[0], Qwen3MoeSparseMoeBlock) or isinstance(
            target.ops[0], Llama4TextExperts
        ):
            if isinstance(target.ops[0], Qwen3MoeSparseMoeBlock):
                experts: list[Qwen3MoeMLP] = target.ops[0].experts
                num_experts = len(experts)
                intermediate_size = experts[0].intermediate_size
                hidden_size = experts[0].hidden_size
                w13 = torch.stack(
                    [
                        torch.cat((e.gate_proj.weight, e.up_proj.weight), dim=0)
                        for e in experts
                    ]
                )
                w2 = torch.stack([e.down_proj.weight for e in experts])
            elif isinstance(target.ops[0], Llama4TextExperts):
                experts: Llama4TextExperts = target.ops[0]
                num_experts = experts.num_experts
                intermediate_size = experts.intermediate_size
                hidden_size = experts.hidden_size
                w13 = experts.gate_up_proj.permute(0, 2, 1).contiguous()
                w2 = experts.down_proj.permute(0, 2, 1).contiguous()

            if qcfg.weight_block_size is not None:
                n, k = qcfg.weight_block_size
                assert hidden_size % n == 0 and hidden_size % k == 0
                assert intermediate_size % n == 0 and intermediate_size % k == 0
            else:
                n, k = 1, 1

            w13 = w13.to(device)
            assert w13.shape == torch.Size(
                [num_experts, 2 * intermediate_size, hidden_size]
            )
            w13 = w13.reshape(num_experts, 2, intermediate_size, hidden_size)
            q13_scale = qcfg.compute_qparams(w13)
            if qcfg.weight_block_size is None:
                assert q13_scale.shape == torch.Size([num_experts, 2]), q13_scale.shape
            else:
                assert q13_scale.shape == torch.Size(
                    [num_experts, 2, intermediate_size // n, hidden_size // k]
                ), (
                    q13_scale.shape,
                    [num_experts, 2, intermediate_size // n, hidden_size // k],
                )

            q13 = qcfg.quantize_tensor(w13, q13_scale).to(qcfg.dtype)
            assert q13.shape == w13.shape, (q13.shape, w13.shape)
            q13 = q13.reshape(num_experts, 2 * intermediate_size, hidden_size)

            w2 = w2.to(device)
            assert w2.shape == torch.Size([num_experts, hidden_size, intermediate_size])
            q2_scale = qcfg.compute_qparams(w2)
            if qcfg.weight_block_size is None:
                assert q2_scale.shape == torch.Size([num_experts])
            else:
                assert q2_scale.shape == torch.Size(
                    [num_experts, hidden_size // n, intermediate_size // k]
                )

            q2 = qcfg.quantize_tensor(w2, q2_scale).to(qcfg.dtype)
            assert q2.shape == w2.shape

            packed = PackedMoELayer(
                q13.cpu(), q2.cpu(), q13_scale.cpu(), q2_scale.cpu()
            )
            set_submodule(model, experts, packed)

        else:
            raise NotImplementedError(target.ops)

    model.config.quantization_config = {
        "quant_method": "fp8",
        "is_checkpoint_fp8_serialized": True,
        "activation_scheme": "dynamic",
        "weight_block_size": (
            None if qcfg.weight_block_size is None else list(qcfg.weight_block_size)
        ),
    }

    model_packed_num_bytes = sum(
        p.numel() * p.dtype.itemsize for p in model.parameters()
    )

    LOGGER.info(
        f"Model compressed from {model_original_num_bytes * 1e-9:.1f} gb to {model_packed_num_bytes * 1e-9:.1f} gb"
    )


class PackedLinearLayer(torch.nn.Module):
    def __init__(
        self,
        weight: torch.Tensor,
        weight_scale: torch.Tensor,
        input_scale: torch.Tensor = None,
    ):
        super().__init__()
        self.weight = torch.nn.Parameter(weight, requires_grad=False)
        self.weight_scale = torch.nn.Parameter(weight_scale, requires_grad=False)
        self.input_scale = None
        if input_scale is not None:
            self.input_scale = torch.nn.Parameter(input_scale, requires_grad=False)

    def forward(self, *args, **kwargs):
        raise NotImplementedError("This class is only used for serialization")


class PackedMoELayer(torch.nn.Module):
    def __init__(
        self,
        w13_weight: torch.Tensor,
        w2_weight: torch.Tensor,
        w13_weight_scale: torch.Tensor,
        w2_weight_scale: torch.Tensor,
        w13_input_scale: torch.Tensor = None,
        w2_input_scale: torch.Tensor = None,
    ):
        super().__init__()
        self.w13_weight = torch.nn.Parameter(w13_weight, requires_grad=False)
        self.w2_weight = torch.nn.Parameter(w2_weight, requires_grad=False)
        self.w13_weight_scale = torch.nn.Parameter(
            w13_weight_scale, requires_grad=False
        )
        self.w2_weight_scale = torch.nn.Parameter(w2_weight_scale, requires_grad=False)
        self.w13_input_scale = None
        if w13_input_scale is not None:
            self.w13_input_scale = torch.nn.Parameter(
                w13_input_scale, requires_grad=False
            )
        self.w2_input_scale = None
        if w2_input_scale is not None:
            self.w2_input_scale = torch.nn.Parameter(
                w2_input_scale, requires_grad=False
            )

    def forward(self, *args, **kwargs):
        raise NotImplementedError("This class is only used for serialization")
