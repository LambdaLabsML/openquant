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
        y = (x.float() / scale).clamp(min=self.min_value, max=self.max_value)
        y = y.to(x.dtype).to(self.dtype)
        y = y.reshape(*shape, N, K)
        return y

    def dequantize_tensor(self, x: torch.Tensor, scale: torch.Tensor):
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
        y = x * scale
        y = y.reshape(*shape, N, K)
        return y

    def compute_scale(self, x: torch.Tensor) -> torch.Tensor:
        assert x.ndim >= 2
        *shape, N, K = x.shape
        if self.weight_block_size is not None:
            n, k = self.weight_block_size
            assert N % n == 0 and K % k == 0
            x = x.reshape(*shape, N // n, n, K // k, k)
            dims_to_reduce = [x.ndim - 3, x.ndim - 1]
        else:
            dims_to_reduce = [x.ndim - 2, x.ndim - 1]
        scale = (x.float().abs().amax(dim=dims_to_reduce) / self.max_value).clamp(
            # NOTE: special value found from vllm https://github.com/vllm-project/vllm/blob/v0.9.0/csrc/quantization/utils.cuh#L50
            min=1.0
            / (512.0 * self.min_value)
        )
        if self.weight_block_size is not None:
            scale = scale.to(x.dtype)
        return scale


@torch.inference_mode()
def pack(qcfg: QuantConfig, model: torch.nn.Module, targets: list[QuantTarget]):
    model_original_num_bytes = sum(
        p.numel() * p.dtype.itemsize for p in model.parameters()
    )

    packed_linear = PackedLinear
    if qcfg.weight_block_size is not None:
        packed_linear = PackedBlockLinear

    ignored_layers = set()
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            ignored_layers.add(name)

    for target in targets:
        for module in target.ops:
            if isinstance(module, torch.nn.Linear):
                w = module.weight
                scale = qcfg.compute_scale(w)
                q = qcfg.quantize_tensor(w, scale)
                packed = packed_linear(q, scale)
                ignored_layers.discard(set_submodule(model, module, packed))
                module.to("meta")

            elif isinstance(module, Qwen3MoeSparseMoeBlock) or isinstance(
                module, Llama4TextExperts
            ):
                if isinstance(module, Qwen3MoeSparseMoeBlock):
                    experts: list[Qwen3MoeMLP] = module.experts
                    num_experts = len(experts)
                    intermediate_size = experts[0].intermediate_size
                    hidden_size = experts[0].hidden_size
                    gates = [e.gate_proj.weight for e in experts]
                    ups = [e.up_proj.weight for e in experts]
                    downs = [e.down_proj.weight for e in experts]
                elif isinstance(module, Llama4TextExperts):
                    experts: Llama4TextExperts = module
                    num_experts = experts.num_experts
                    intermediate_size = experts.intermediate_size
                    hidden_size = experts.hidden_size
                    gu = experts.gate_up_proj.permute(0, 2, 1).reshape(
                        num_experts, 2, intermediate_size, hidden_size
                    )
                    gate, up = gu.unbind(dim=1)
                    down = experts.down_proj.permute(0, 2, 1)
                    gates = [g for g in gate.unbind()]
                    ups = [u for u in up.unbind()]
                    downs = [d for d in down.unbind()]

                assert all(
                    g.shape == torch.Size([intermediate_size, hidden_size])
                    for g in gates
                )
                assert all(
                    u.shape == torch.Size([intermediate_size, hidden_size]) for u in ups
                )
                assert all(
                    d.shape == torch.Size([hidden_size, intermediate_size])
                    for d in downs
                )
                if qcfg.weight_block_size is not None:
                    n, k = qcfg.weight_block_size
                    assert hidden_size % n == 0 and hidden_size % k == 0
                    assert intermediate_size % n == 0 and intermediate_size % k == 0
                else:
                    n, k = 1, 1

                packed_experts = []
                for gate, up, down in zip(gates, ups, downs):
                    gate_scale = qcfg.compute_scale(gate)
                    gate_q = qcfg.quantize_tensor(gate, gate_scale)

                    up_scale = qcfg.compute_scale(up)
                    up_q = qcfg.quantize_tensor(up, up_scale)

                    down_scale = qcfg.compute_scale(down)
                    down_q = qcfg.quantize_tensor(down, down_scale)

                    packed_experts.append(
                        PackedMlp(
                            down_proj=packed_linear(down_q, down_scale),
                            gate_proj=packed_linear(gate_q, gate_scale),
                            up_proj=packed_linear(up_q, up_scale),
                        )
                    )

                name = set_submodule(
                    model, experts, torch.nn.ModuleList(packed_experts)
                )
                ignored_layers = {l for l in ignored_layers if not l.startswith(name)}

                experts.to("meta")

            else:
                raise NotImplementedError(target.ops)

    model.config.quantization_config = {
        "quant_method": "fp8",
        "is_checkpoint_fp8_serialized": True,
        "activation_scheme": "dynamic",
        "weight_block_size": (
            None if qcfg.weight_block_size is None else list(qcfg.weight_block_size)
        ),
        "ignored_layers": list(ignored_layers),
    }

    model_packed_num_bytes = sum(
        p.numel() * p.dtype.itemsize for p in model.parameters()
    )

    LOGGER.info(f"Ignored {ignored_layers}")
    LOGGER.info(
        f"Model compressed from {model_original_num_bytes * 1e-9:.1f} gb to {model_packed_num_bytes * 1e-9:.1f} gb"
    )


class PackedLinear(torch.nn.Module):
    def __init__(
        self,
        weight: torch.Tensor,
        weight_scale: torch.Tensor,
    ):
        super().__init__()
        self.weight = torch.nn.Parameter(weight, requires_grad=False)
        self.weight_scale = torch.nn.Parameter(weight_scale, requires_grad=False)

    def forward(self, *args, **kwargs):
        raise NotImplementedError("This class is only used for serialization")


class PackedBlockLinear(torch.nn.Module):
    def __init__(
        self,
        weight: torch.Tensor,
        weight_scale: torch.Tensor,
    ):
        super().__init__()
        self.weight = torch.nn.Parameter(weight, requires_grad=False)
        # NOTE: no idea why this is called weight_scale_inv, they are treated the same,
        #       so notably it is NOT 1 / weight_scale
        self.weight_scale_inv = torch.nn.Parameter(weight_scale, requires_grad=False)

    def forward(self, *args, **kwargs):
        raise NotImplementedError("This class is only used for serialization")


class PackedMlp(torch.nn.Module):
    def __init__(
        self,
        down_proj: torch.nn.Module,
        gate_proj: torch.nn.Module,
        up_proj: torch.nn.Module,
    ):
        super().__init__()
        self.down_proj = down_proj
        self.gate_proj = gate_proj
        self.up_proj = up_proj

    def forward(self, *args, **kwargs):
        raise NotImplementedError("This class is only used for serialization")
