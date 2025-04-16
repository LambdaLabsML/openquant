import argparse
import os
import logging

import tqdm
import torch
import datasets
from transformers import AutoTokenizer, AutoModelForCausalLM, default_data_collator
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.modeling_utils import PreTrainedModel
from accelerate import cpu_offload

from openquant import (
    LinearQuantizer,
    AWQ,
    QuantConfig,
    GraphTracer,
    ForwardPassEarlyStop,
)

LOGGER = logging.getLogger(__name__)


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "-q",
        "--quantization",
        default=None,
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
    parser.add_argument("--cpu-offload", default=False, action="store_true")
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

    tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(
        args.model, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def preprocess(example):
        return {
            "text": tokenizer.apply_chat_template(example["messages"], tokenize=False)
        }

    def tokenize(sample):
        return tokenizer(
            sample["text"],
            padding="max_length",
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

    if args.cpu_offload:
        model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
            args.model, attn_implementation=attn_implementation
        )
        cpu_offload(model, execution_device=device)
    else:
        with device:
            model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
                args.model, attn_implementation=attn_implementation
            )

    quant_cls: type[LinearQuantizer]
    if args.quantization == "AWQ-Int4":
        quant_config = QuantConfig(num_bits=4)
        quant_cls = AWQ
    # elif args.quantization == "GPTQ-Int4":
    #     quant_config = QuantConfig(num_bits=4)
    #     quant_cls = GPTQ
    # elif args.quantization == "GPTQ-Int8":
    #     quant_config = QuantConfig(num_bits=8)
    #     quant_cls = GPTQ
    # elif args.quantization == "Dynamic-F8":
    #     quant_config = QuantConfig(num_bits=8)
    #     quant_cls = DynamicFloat
    # elif args.quantization == "Static-F8":
    #     quant_config = QuantConfig(num_bits=8)
    #     quant_cls = StaticFloat
    # elif args.quantization == "Dynamic-F4":
    #     quant_config = QuantConfig(num_bits=4)
    #     quant_cls = DynamicFloat
    # elif args.quantization == "Static-F4":
    #     quant_config = QuantConfig(num_bits=4)
    #     quant_cls = StaticFloat
    else:
        raise NotImplementedError(args.quantization)

    LOGGER.info(
        f"Using {quant_config}. Value range is: [{quant_config.min_int}, {quant_config.max_int}]"
    )

    LOGGER.info("Setting up graph tracing")
    GraphTracer.init_hooks(model)

    target_modules: dict[str, torch.nn.Module] = {}
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            LOGGER.debug(f"Targeting {name}")
            target_modules[name] = module

    pbar = tqdm.tqdm(total=len(target_modules))
    for name, module in target_modules.items():
        assert len(name) > 0
        assert not isinstance(module, GraphTracer)

        pbar.set_description(f"Quantizing {name}")

        quantizer: LinearQuantizer = quant_cls(
            quant_config,
            name,
            module,
            execution_device=device,
            storage_device=torch.device("cpu"),
        )
        module.register_forward_pre_hook(quantizer.pre_forward_hook)

        GraphTracer.clear()

        # calibrate
        for i in tqdm.tqdm(
            range(0, len(ds), args.batch_size), leave=False, desc="Fwd Pass Calibration"
        ):
            uncollated_batch = ds[i : i + args.batch_size]
            collated_batch = default_data_collator(uncollated_batch)
            model_inputs = model.prepare_inputs_for_generation(**collated_batch)
            try:
                _ = model(**model_inputs)
            except ForwardPassEarlyStop:
                pass

        # do quantization
        # TODO do we have to re-wrap this with GraphTracer?
        quantized_module = quantizer.quantize_module()
        assert quantized_module is not None
        setattr(model, name, quantized_module)

    LOGGER.info(f"Saving quantized model to {quant_name}")
    model.config.quantization_config = quant_cls.get_transformers_quant_config(
        quant_config
    )
    model.save_pretrained(quant_name)
    tokenizer.save_pretrained(quant_name)


if __name__ == "__main__":
    main()
