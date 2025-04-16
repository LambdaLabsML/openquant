import math
import argparse
import os
import logging
import weakref

import tqdm
import torch
import torch.nn.functional as F
import huggingface_hub
import datasets
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, default_data_collator
from transformers.modeling_utils import PreTrainedModel
from accelerate import cpu_offload


LOGGER = logging.getLogger(__name__)


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


class GPTQ(LinearQuantizer):
    def __init__(
        self,
        qcfg: QuantConfig,
        name: str,
        module: torch.nn.Linear,
        execution_device: torch.device,
        cache_device: torch.device,
        damp_percent: float = 0.01,
        damp_auto_increment: float = 0.0025,
        static_groups: bool = False,
    ):
        super().__init__(qcfg, name, module, execution_device, cache_device)

        assert 0 < damp_percent < 1
        assert 0 < damp_auto_increment < 1
        self.damp_percent = damp_percent
        self.damp_auto_increment = damp_auto_increment
        self.static_groups = static_groups

        self.h: torch.Tensor = torch.zeros(
            (self.inp_dim, self.inp_dim),
            device=cache_device,
            requires_grad=False,
        )
        self.num_samples = 0

    @torch.inference_mode()
    def forward(self, x: torch.Tensor):
        batch_size, _inp_dim = x.shape

        x = x.to(self.execution_device)
        h = self.h.to(self.execution_device)

        h *= self.num_samples / (self.num_samples + batch_size)
        self.num_samples += batch_size
        x = math.sqrt(2 / self.num_samples) * x.float()
        h += x.t().matmul(x)

        self.h = h.to(self.cache_device)
        x.cpu()

        raise ForwardPassEarlyStop()

    @torch.inference_mode()
    def quantize_module(self):
        w = self.module.weight.data.clone().to(self.execution_device)
        h = self.h.to(self.execution_device)
        q = torch.zeros_like(w)
        scale, zero = self.qcfg.compute_qparams(w)

        dead = torch.diag(h) == 0
        h[dead, dead] = 1
        w[:, dead] = 0

        perm = torch.argsort(torch.diag(h), descending=True)
        invperm = torch.argsort(perm)
        w = w[:, perm]
        h = h[perm][:, perm]

        losses = torch.zeros_like(w)

        diag = torch.arange(self.columns, device=self.execution_device)
        damp_percent = self.qcfg.damp_percent
        while 0 < damp_percent < 1:
            try:
                h[diag, diag] += damp_percent * torch.mean(torch.diag(h))
                h = torch.cholesky(h)
                h = torch.cholesky_inverse(h)
                h = torch.cholesky(h, upper=True)
                h_inverse = h
                break
            except torch._C._LinAlgError as e:
                damp_percent += self.qcfg.damp_auto_increment
                assert 0 < damp_percent < 1

        for i1 in range(0, self.inp_dim, self.qcfg.group_size):
            i2 = min(i1 + self.qcfg.group_size, self.inp_dim)
            count = i2 - i1

            for i in range(count):
                local_w = w[:, i1 + i : i1 + i + 1]
                local_q = self.qcfg.quantize_tensor(local_w, scale, zero).flatten()

                d = h_inverse1[i, i]

                if (i1 + i) % self.qcfg.group_size == 0:
                    self.qcfg.compute_qparams(
                        w[:, (i1 + i) : (i1 + i + self.qcfg.group_size)],
                        weight=True,
                    )

                if ((i1 + i) // self.qcfg.group_size) - now_idx == -1:
                    scale.append(self.quantizer.scale)
                    zero.append(self.quantizer.zero)
                    now_idx += 1

                q1[:, i] = q
                losses1[:, i] = (w - q) ** 2 / d**2

                err1 = (w - q) / d
                W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
                Err1[:, i] = err1

            Q[:, i1:i2] = Q1
            Losses[:, i1:i2] = Losses1 / 2

            W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])

        return q, scale, zero


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


class DynamicFloat(LinearQuantizer):
    pass


class StaticFloat(LinearQuantizer):
    pass


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "-q",
        "--quantization",
        choices=[
            "AWQ-Int4",
            "GPTQ-Int8",
            "GPTQ-Int4",
            "Static-F8",
            "Dynamic-F8",
        ],
        required=True,
        help="The type of quantization to apply",
    )
    parser.add_argument(
        "-m",
        "--model",
        default=None,
        required=True,
        help="The base model. Should be huggingface tag.",
    )
    parser.add_argument(
        "--dataset", default="HuggingFaceH4/ultrachat_200k", help="Calibration data"
    )
    parser.add_argument(
        "--dataset-split", default="train_sft", help="Split for calibration data"
    )
    parser.add_argument(
        "--dataset-name",
        default=None,
        help="Name for calibration data, passed to datasets.load_dataset.",
    )
    parser.add_argument(
        "--num-samples",
        default=512,
        type=int,
        help="Number of items from dataset to use for calibration",
    )
    parser.add_argument(
        "--seq-length",
        default=2048,
        type=int,
        help="Sequence length for calibration data",
    )
    parser.add_argument(
        "--batch-size",
        default=32,
        type=int,
        help="Number of calibration samples to process at the same time.",
    )
    args = parser.parse_args()

    model_name = os.path.basename(args.model)
    quant_name = f"{model_name}-{args.quantization}"

    logging.basicConfig(level=logging.INFO)

    LOGGER.info(args)
    LOGGER.info(os.environ)

    target_device = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]
    LOGGER.info(f"Using cuda:{target_device}")

    # NOTE: `0` index means the first **visible** cuda device
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    LOGGER.info(f"Saving quantized model to {quant_name}")

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    def preprocess(example):
        return {
            "text": tokenizer.apply_chat_template(example["messages"], tokenize=False)
        }

    def tokenize(sample):
        return tokenizer(
            sample["text"],
            padding=False,
            max_length=args.seq_length,
            truncation=True,
            add_special_tokens=False,
        )

    ds = datasets.load_dataset(
        args.dataset, args.dataset_name, split=args.dataset_split
    )
    ds = ds.shuffle(seed=0).select(range(args.num_samples))
    ds = ds.map(preprocess)
    ds = ds.map(tokenize, remove_columns=ds.column_names)
    ds = ds.to_list()

    try:
        import flash_attn

        attn_implementation = "flash_attention_2"
        LOGGER.info(f"Using flash attention")
    except ImportError:
        attn_implementation = None
        LOGGER.info(f"`import flash_attn` not found, pip install to use")

    model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
        args.model, attn_implementation=attn_implementation
    )
    cpu_offload(model, execution_device=device)

    quant_cls: type[LinearQuantizer]
    if args.quantization == "AWQ-Int4":
        quant_config = QuantConfig(num_bits=4)
        quant_cls = AWQ
    elif args.quantization == "GPTQ-Int4":
        quant_config = QuantConfig(num_bits=4)
        quant_cls = GPTQ
    elif args.quantization == "GPTQ-Int8":
        quant_config = QuantConfig(num_bits=8)
        quant_cls = GPTQ
    elif args.quantization == "Dynamic-F8":
        quant_config = QuantConfig(num_bits=8)
        quant_cls = DynamicFloat
    elif args.quantization == "Static-F8":
        quant_config = QuantConfig(num_bits=8)
        quant_cls = StaticFloat
    elif args.quantization == "Dynamic-F4":
        quant_config = QuantConfig(num_bits=4)
        quant_cls = DynamicFloat
    elif args.quantization == "Static-F4":
        quant_config = QuantConfig(num_bits=4)
        quant_cls = StaticFloat
    else:
        raise NotImplementedError(args.quantization)

    LOGGER.info(
        f"Using QuantConfig(num_bits={quant_config.num_bits}). Value range is: [{quant_config.min_int}, {quant_config.max_int}]"
    )

    LOGGER.info("Setting up graph tracing")
    for name, module in model.named_modules():
        setattr(model, name, GraphTracer(name, module))

    target_modules = {}
    for name, module in model.named_modules():
        assert isinstance(module, GraphTracer)
        module = module.module
        if isinstance(module, torch.nn.Linear):
            LOGGER.debug(f"Targeting {name}")
            target_modules[name] = module

    pbar = tqdm.tqdm(total=len(target_modules))
    for name, module in target_modules.items():
        assert not isinstance(module, GraphTracer)

        pbar.set_description(name)

        quantizer: LinearQuantizer = quant_cls(
            quant_config,
            module,
            execution_device=device,
            storage_device=torch.device("cpu"),
        )
        setattr(model, name, quantizer)

        GraphTracer.clear()

        # calibrate
        for i in tqdm.tqdm(
            range(0, len(ds), args.batch_size), leave=False, desc="Fwd Pass Calibration"
        ):
            uncollated_batch = ds[i : i + args.batch_size]
            collated_batch = default_data_collator(uncollated_batch)
            model_inputs = model.prepare_inputs_for_generation(collated_batch)
            try:
                _ = model.generate(model_inputs, max_new_tokens=1)
            except ForwardPassEarlyStop:
                pass

        # do quantization
        # TODO do we have to re-wrap this with GraphTracer?
        quantized_module = quantizer.quantize_module()
        setattr(model, name, quantized_module)

    model.config.quantization_config = quant_cls.get_transformers_quant_config(
        quant_config
    )
    model.save_pretrained(quant_name)
    tokenizer.save_pretrained(quant_name)


if __name__ == "__main__":
    main()
