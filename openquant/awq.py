import logging

import tqdm
import torch
import torch.nn.functional as F
import torch.distributed as dist

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
        # TODO support num_bits = 8 - order_map for packing is [0, 2, 1, 3]
        if num_bits != 4:
            raise NotImplementedError(
                "AWQ num_bits only supports 4 bits, packing cannot be done with other sizes"
            )
        if group_size not in [32, 64, 128]:
            raise NotImplementedError("AWQ group_size must be one of [32, 64, 128]")

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

    def qdq_tensor(self, x: torch.Tensor, scale: torch.Tensor, zero: torch.Tensor):
        assert x.ndim == 2
        shape = x.shape
        x = x.reshape(shape[0], shape[1] // self.group_size, self.group_size)
        scale = scale.unsqueeze(-1)
        zero = zero.unsqueeze(-1)

        # NOTE: quantize
        x = torch.clamp(torch.round(x / scale) + zero, self.min_int, self.max_int)
        # NOTE: dequantize
        x = (x - zero) * scale

        return x.reshape(shape)

    def quantize_tensor(self, x: torch.Tensor, scale: torch.Tensor, zero: torch.Tensor):
        assert x.ndim == 2
        shape = x.shape
        x = x.reshape(shape[0], shape[1] // self.group_size, self.group_size)
        scale = scale.unsqueeze(-1)
        zero = zero.unsqueeze(-1)

        # NOTE: quantize
        x = torch.clamp(torch.round(x / scale) + zero, self.min_int, self.max_int)

        return x.reshape(shape)

    def compute_qparams(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        assert x.ndim == 2 and x.shape[-1] % self.group_size == 0
        shape = x.shape

        x = x.reshape(-1, self.group_size)

        # TODO use quantile instead of amin/amax?
        if self.zero_point:
            min_val = x.amin(dim=1)
            max_val = x.amax(dim=1)
            scale = (max_val - min_val) / self.max_int
            zero = (-torch.round(min_val / scale)).clamp(self.min_int, self.max_int)
        else:
            max_val = x.abs().amax(dim=1)
            scale = max_val / self.max_int
            zero = torch.zeros_like(max_val)

        return scale.reshape(shape[0], -1), zero.reshape(shape[0], -1)


@torch.inference_mode()
def quantize(
    qcfg: QuantConfig,
    target: QuantTarget,
    inputs,
    device: torch.device,
    search_grid_size: int = 20,
    enable_early_stopping: bool = True,
):
    """
    AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration

    https://arxiv.org/abs/2306.00978
    """
    clean_memory(device)
    assert all(isinstance(m, torch.nn.Linear) for m in target.ops)

    inp_dim = target.ops[0].in_features
    assert inp_dim % qcfg.group_size == 0
    assert all(m.in_features == inp_dim for m in target.ops)

    xs: list[torch.Tensor] = []
    for args, kwargs in tqdm.tqdm(
        inputs,
        desc="Reshaping inputs",
        leave=False,
        disable=dist.is_initialized(),
    ):
        assert len(args) == 1, len(args)
        assert isinstance(args[0], torch.Tensor)
        assert args[0].shape[-1] == inp_dim, args[0].shape
        xs.append(args[0].reshape(-1, inp_dim))

    total_batch_size = sum(x.shape[0] for x in xs)
    assert total_batch_size > 0

    w = torch.cat([m.weight for m in target.ops]).to(device)
    total_out_size = sum(m.out_features for m in target.ops)
    assert w.shape[0] == total_out_size

    # NOTE: input magnitude calculation (per input channel i.e. per `inp_dim`)
    s_x: torch.Tensor = 0
    # TODO filter out padded!!!
    for x in tqdm.tqdm(
        xs,
        desc="Computing input mangitude",
        leave=False,
        disable=dist.is_initialized(),
    ):
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
    best_scale = None
    best_zero = None
    for alpha in tqdm.tqdm(
        torch.linspace(0, 1, steps=search_grid_size + 1),
        desc="Searching scales",
        leave=False,
        total=search_grid_size + 1,
        disable=dist.is_initialized(),
    ):
        s = s_x.pow(alpha).clamp(min=1e-8)
        s[torch.isinf(s) | torch.isnan(s)] = 1

        w_s = w * s
        scale, zero = qcfg.compute_qparams(w_s)
        w_qdq = qcfg.qdq_tensor(w_s, scale, zero)

        loss = 0
        for x in xs:
            x = x.to(device)
            # NOTE: don't need to use bias because the bias will be unchanged and so will cancel each other out
            y = F.linear(x, w).float()
            y_qdq = F.linear(x / s, w_qdq).float()
            loss += (y - y_qdq).square().sum()
        loss /= total_batch_size * total_out_size
        LOGGER.debug(f"loss={loss} @ alpha={alpha:.3f}")

        if loss < best_loss:
            best_loss = loss
            best_s = s.to("cpu", non_blocking=True)
            best_alpha = alpha
            best_scale = scale.to("cpu", non_blocking=True)
            best_zero = zero.to("cpu", non_blocking=True)
        elif enable_early_stopping:
            # NOTE: the loss **tends to** decrease until the minimum, and then increase,
            #       so once the loss starts to increase we just stop early
            break

    LOGGER.info(f"best_loss={best_loss} @ alpha={best_alpha:.3f}")

    scales = torch.split(best_scale, [m.out_features for m in target.ops], dim=0)
    zeros = torch.split(best_zero, [m.out_features for m in target.ops], dim=0)
    return best_s, scales, zeros


@torch.inference_mode()
def pack(
    qcfg: QuantConfig,
    model: torch.nn.Module,
    targets: list[
        tuple[QuantTarget, torch.Tensor, list[torch.Tensor], list[torch.Tensor]]
    ],
    pack_dtype=torch.int32,
):
    if qcfg.num_bits == 4:
        # NOTE: not sure why, but you can see this used in VLLM marlin kernels here
        # https://github.com/vllm-project/vllm/blob/v0.8.5.post1/vllm/model_executor/layers/quantization/utils/marlin_utils.py#L257
        order_map = [0, 2, 4, 6, 1, 3, 5, 7]
    else:
        raise NotImplementedError("Only 4-bit are supported for now.")

    pack_dtype_num_bits = pack_dtype.itemsize * 8
    assert pack_dtype_num_bits % qcfg.num_bits == 0
    pack_factor = pack_dtype_num_bits // qcfg.num_bits
    assert pack_factor == len(order_map)
    LOGGER.info(f"Packing {pack_factor} int{qcfg.num_bits} into {pack_dtype}")

    for target, best_s, _, _ in tqdm.tqdm(
        targets,
        desc="Scaling parent ops",
        leave=False,
        disable=dist.is_initialized(),
    ):
        for module in target.ops:
            assert isinstance(module, torch.nn.Linear)
            assert module.in_features == target.ops[0].in_features

        inp_dim = target.ops[0].in_features
        parent = target.parent
        parent_op_name = parent.__class__.__name__.lower()
        if isinstance(parent, torch.nn.Linear):
            if parent.weight.shape[0] == inp_dim:
                parent.weight.div_(best_s.view(inp_dim, 1))
            elif parent.weight.shape[1] == inp_dim:
                parent.weight.div_(best_s.view(1, inp_dim))
            else:
                raise ValueError(
                    f"Tried scaling parent op with different inp/out dimensions. inp_dim={inp_dim} parent_shape={parent.weight.shape}"
                )

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

    model_original_num_bytes = sum(
        p.numel() * p.dtype.itemsize for p in model.parameters()
    )

    for target, best_s, scales, zero in targets:
        for module, scale, zero in zip(target.ops, scales, zero):
            assert isinstance(module, torch.nn.Linear)

            w_q = qcfg.quantize_tensor(module.weight.data * best_s, scale, zero)
            assert w_q.shape == torch.Size([module.out_features, module.in_features])

            qweight = torch.zeros(
                module.in_features,
                module.out_features // pack_factor,
                dtype=pack_dtype,
            )
            for col in range(module.out_features // pack_factor):
                for i in range(pack_factor):
                    qcol = w_q[col * pack_factor + order_map[i], :].to(pack_dtype)
                    qweight[:, col] |= qcol << (i * qcfg.num_bits)

            assert zero.shape == torch.Size(
                [module.out_features, module.in_features // qcfg.group_size]
            )
            qzeros = torch.zeros(
                module.in_features // qcfg.group_size,
                module.out_features // pack_factor,
                dtype=pack_dtype,
            )
            for col in range(module.out_features // pack_factor):
                for i in range(pack_factor):
                    qcol = zero[col * pack_factor + order_map[i], :].to(pack_dtype)
                    qzeros[:, col] |= qcol << (i * qcfg.num_bits)

            scale = scale.t().contiguous()
            assert scale.shape == torch.Size(
                [module.in_features // qcfg.group_size, module.out_features]
            )
            assert scale.dtype in [torch.float16, torch.bfloat16]

            # NOTE: for vllm compatibility with tensor shapes/dtypes check here:
            # https://github.com/vllm-project/vllm/blob/v0.8.5.post1/vllm/model_executor/layers/quantization/awq.py#L98
            packed = PackedLayer(qweight, qzeros, scale, module.bias)

            module_name = None
            for name, haystack in model.named_modules():
                if haystack == module:
                    module_name = name
                    break
            assert module_name is not None

            # NOTE: we have to get the owner this way because torch stores list index as `.<index>.<next part>`,
            # and that is not compatible with getattr & lists
            owner = model
            name_parts = module_name.split(".")
            for attr in name_parts[:-1]:
                if attr.isnumeric():
                    owner = owner[int(attr)]
                else:
                    owner = getattr(owner, attr)
            setattr(owner, name_parts[-1], packed)

            original_num_bytes = sum(
                p.numel() * p.dtype.itemsize for p in module.parameters()
            )
            packed_num_bytes = sum(
                p.numel() * p.dtype.itemsize for p in packed.parameters()
            )
            LOGGER.info(
                f"Packed {module_name} from {original_num_bytes * 1e-6:.1f} mb to {packed_num_bytes * 1e-6:.1f} mb"
            )

    model_packed_num_bytes = sum(
        p.numel() * p.dtype.itemsize for p in model.parameters()
    )

    LOGGER.info(
        f"Model compressed from {model_original_num_bytes * 1e-9:.1f} gb to {model_packed_num_bytes * 1e-9:.1f} gb"
    )

    # NOTE: vllm compatibility here
    # https://github.com/vllm-project/vllm/blob/v0.8.5.post1/vllm/model_executor/layers/quantization/awq.py#L67
    model.config.quantization_config = {
        "quant_method": "awq",
        "zero_point": qcfg.zero_point,
        "group_size": qcfg.group_size,
        "bits": qcfg.num_bits,
        "version": "gemm",
        "modules_to_not_convert": None,
    }


class PackedLayer(torch.nn.Module):
    def __init__(
        self,
        qweight: torch.Tensor,
        qzeros: torch.Tensor,
        scales: torch.Tensor,
        bias: torch.Tensor,
    ):
        super().__init__()
        self.qweight = torch.nn.Parameter(qweight, requires_grad=False)
        self.qzeros = torch.nn.Parameter(qzeros, requires_grad=False)
        self.scales = torch.nn.Parameter(scales, requires_grad=False)
        self.bias = torch.nn.Parameter(bias, requires_grad=False)

    def forward(self, *args, **kwargs):
        raise NotImplementedError("This class is only used for serialization")
