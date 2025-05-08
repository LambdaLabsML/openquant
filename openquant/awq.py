import logging

import tqdm
import torch
import torch.nn.functional as F

from . import clean_memory
from .subgraph import QuantTarget

LOGGER = logging.getLogger(__name__)


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

    def quantize_tensor(self, x: torch.Tensor, scale: torch.Tensor, zero: torch.Tensor):
        x = torch.clamp(torch.round(x / scale) + zero, self.min_int, self.max_int)
        return (x - zero) * scale

    def dequantize_tensor(
        self, x: torch.Tensor, scale: torch.Tensor, zero: torch.Tensor
    ):
        return (x - zero) * scale

    def compute_qparams(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        shape = x.shape

        x = x.reshape(-1, self.group_size)

        # TODO use quantile instead of amin/amax?
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


@torch.inference_mode()
def quantize(
    qcfg: QuantConfig,
    target: QuantTarget,
    inputs,
    device: torch.device,
    search_grid_size: int = 20,
):
    """
    AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration

    https://arxiv.org/abs/2306.00978
    """
    modules_to_scale: list[torch.nn.Linear] = target.ops
    module_to_inverse_scale: torch.nn.Module = target.parent

    inp_dim = modules_to_scale[0].in_features
    assert inp_dim % qcfg.group_size == 0
    assert all(m.in_features == inp_dim for m in modules_to_scale)

    xs: list[torch.Tensor] = []
    for args, kwargs in tqdm.tqdm(inputs, desc="Reshaping inputs", leave=False):
        assert len(args) == 1, len(args)
        assert isinstance(args[0], torch.Tensor)
        assert args[0].shape[-1] == inp_dim, args[0].shape
        xs.append(args[0].reshape(-1, inp_dim))

    total_batch_size = sum(x.shape[0] for x in xs)
    assert total_batch_size > 0

    w = torch.cat([m.weight for m in modules_to_scale]).to(device)

    # NOTE: input magnitude calculation (per input channel i.e. per `inp_dim`)
    s_x: torch.Tensor = 0
    # TODO filter out padded!!!
    for x in tqdm.tqdm(xs, desc="Computing input mangitude", leave=False):
        # NOTE: x has shape [batch, inp_dim]
        # NOTE: this reshape is the group-wise functionality
        g_x = x.to(device).reshape(x.shape[0], -1, qcfg.group_size)
        s_x += g_x.abs().sum(0) / total_batch_size
    s_x = s_x.reshape(inp_dim)

    clean_memory(device)

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

        loss_factor = 1 / (total_batch_size * w.shape[0])
        loss = 0
        for x in xs:
            x = x.to(device)
            # NOTE: don't need to use bias because the bias will be unchanged and so will cancel each other out
            y = F.linear(x, w)
            y_q = F.linear(x / s, w_q)
            loss += ((y - y_q).square() * loss_factor).sum()

        if loss < best_loss:
            best_loss = loss
            best_s = s
            best_alpha = alpha

        pbar.set_description(
            f"Searching scales: best_loss={best_loss} @ alpha={best_alpha:.3f}",
            refresh=False,
        )
        pbar.update()
    pbar.close()

    LOGGER.info(f"best_loss={best_loss} @ alpha={best_alpha:.3f}")

    # Apply scale to module
    scales = []
    zeros = []
    for module in tqdm.tqdm(modules_to_scale, desc="Scaling modules", leave=False):
        w = module.weight.data.to(device)
        w_s = w * best_s
        scale, zero = qcfg.compute_qparams(w_s)
        w_q = qcfg.quantize_tensor(w_s, scale, zero)
        module.weight.data = w_q.to(module.weight.device)

        scales.append(scale.cpu())
        zeros.append(zero.cpu())

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

    return list(zip(modules_to_scale, scales, zeros))


def transformers_quant_config(qcfg: QuantConfig) -> dict:
    return {
        "quant_method": "awq",
        "zero_point": qcfg.zero_point,
        "group_size": qcfg.group_size,
        "bits": qcfg.num_bits,
        "version": "gemm",
        "modules_to_not_convert": None,
    }


def pack(
    qcfg: QuantConfig,
    model: torch.nn.Module,
    targets: list[tuple[torch.nn.Linear, torch.Tensor, torch.Tensor]],
    pack_dtype=torch.int32,
):
    pack_num_bits = pack_dtype.itemsize * 8
    num_packed = pack_num_bits // qcfg.num_bits

    named_targets = []
    for target, scale, zero in targets:
        for name, haystack in model.named_modules():
            if haystack == target:
                named_targets.append((name, target, scale, zero))
                break

    # reference code: https://github.com/casper-hansen/AutoAWQ/blob/main/awq/modules/linear/gemm.py#L172
    for name, target, scale, zero in tqdm.tqdm(
        named_targets, desc="Packing quantized layers", leave=False
    ):
        assert isinstance(target, torch.nn.Linear)

        scaled_zeros: torch.Tensor = zero * scale
        assert scaled_zeros.shape == [
            target.in_features // qcfg.group_size
        ], scaled_zeros.shape

        intweight = []
        for idx in range(target.in_features):
            intweight.append(
                torch.round(
                    (target.weight.data[:, idx] + scaled_zeros[idx // qcfg.group_size])
                    / scale[idx // qcfg.group_size]
                ).to(torch.int)[:, None]
            )
        intweight = torch.cat(intweight, dim=1)
        intweight = intweight.t().contiguous()
        intweight = intweight.to(pack_dtype)

        qweight = torch.zeros(
            (
                intweight.shape[0],
                intweight.shape[1] // pack_num_bits * qcfg.num_bits,
            ),
            dtype=pack_dtype,
            device=target.weight.device,
        )
        for col in range(intweight.shape[1] // num_packed):
            if qcfg.num_bits == 4:
                order_map = [0, 2, 4, 6, 1, 3, 5, 7]
            else:
                raise NotImplementedError("Only 4-bit are supported for now.")
            for i in range(num_packed):
                qweight_col = intweight[:, col * num_packed + order_map[i]]
                qweight[:, col] |= qweight_col << (i * qcfg.num_bits)

        zeros = zeros.to(dtype=pack_dtype)
        qzeros = torch.zeros(
            (zeros.shape[0], zeros.shape[1] // pack_num_bits * qcfg.num_bits),
            dtype=torch.int32,
            device=zeros.device,
        )

        for col in range(zeros.shape[1] // num_packed):
            if qcfg.num_bits == 4:
                order_map = [0, 2, 4, 6, 1, 3, 5, 7]
            else:
                raise NotImplementedError("Only 4-bit are supported for now.")
            for i in range(num_packed):
                qzero_col = zeros[:, col * num_packed + order_map[i]]
                qzeros[:, col] |= qzero_col << (i * qcfg.num_bits)

        setattr(
            model,
            name,
            QuantizedLinear(
                qweight,
                qzeros,
                scale.half(),
                None if target.bias is None else target.bias.half(),
            ),
        )

    raise NotImplementedError()


class QuantizedLinear(torch.nn.Module):
    def __init__(
        self,
        qweight: torch.Tensor,
        qzeros: torch.Tensor,
        scales: torch.Tensor,
        bias: torch.Tensor = None,
    ):
        super().__init__()
        self.register_buffer("qweight", qweight)
        self.register_buffer("qzeros", qzeros)
        self.register_buffer("scales", scales)
        if bias is not None:
            self.register_buffer("bias", bias)
        else:
            self.bias = None

    def forward(self, *args, **kwargs):
        raise NotImplementedError("This class is only used for serialization")
