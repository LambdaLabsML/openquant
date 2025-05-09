import argparse
import os
import logging

import tqdm
import torch
import datasets
from transformers import AutoTokenizer, AutoModelForCausalLM

from openquant import *
from openquant import awq, models

LOGGER = logging.getLogger(__name__)


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "-m",
        "--model",
        help="The base model. Should be huggingface tag.",
        required=True,
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
    parser.add_argument(
        "--no-zero-point",
        action="store_true",
        default=False,
        help="Disable zero-point quantization",
    )
    args = parser.parse_args()

    model_name = os.path.basename(args.model)
    quant_name = f"{model_name}-AWQ-Int4"

    logging.basicConfig(level=logging.INFO)

    LOGGER.info(args)
    LOGGER.info(os.environ)

    target_device = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]
    LOGGER.info(f"Using cuda:{target_device}")

    # NOTE: `0` index means the first **visible** cuda device
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    quant_config = awq.QuantConfig(num_bits=4, zero_point=not args.no_zero_point)
    LOGGER.info(
        f"Using {quant_config}. Value range is: [{quant_config.min_int}, {quant_config.max_int}]"
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
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

    LOGGER.info(f"Loading dataset {args.dataset}")
    ds = datasets.load_dataset(
        args.dataset, args.dataset_name, split=args.dataset_split
    )
    ds = ds.shuffle(seed=0).select(range(args.num_samples))
    ds = ds.map(preprocess)
    ds = ds.map(tokenize, remove_columns=ds.column_names)
    ds = ds.to_list()

    LOGGER.info(f"Loading {args.model}")
    attn_impl = get_attn_implementation()

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        attn_implementation=attn_impl,
        torch_dtype="auto",
        use_cache=False,
    )

    plan = models.make_plan(model)
    for i, target in enumerate(plan):
        LOGGER.info(f"{i+1}. {target.names(model)}")

    last_subgraph = None
    packs = []
    for target in tqdm.tqdm(plan):
        # compute new inputs for this subgraph
        if last_subgraph is None:
            models.head_to_device(model, device)
            subgraph_inputs = init_subgraph_inputs(
                model, target.subgraph, ds, args.batch_size, device
            )
            model.cpu()
        elif last_subgraph != target.subgraph:
            last_subgraph.to(device)
            update_subgraph_inputs(last_subgraph, subgraph_inputs)
            last_subgraph.cpu()

        LOGGER.info(f"Quantizing {target.names(model)}")
        target.subgraph.to(device)

        # get inputs to target
        target_inputs = get_layer_inputs(target.ops, target.subgraph, subgraph_inputs)

        # quantize it
        try:
            packs.append(awq.quantize(quant_config, target, target_inputs, device))
        except torch.OutOfMemoryError:
            LOGGER.error("Sending subgraph back to CPU - reduce batch size to avoid")
            target.subgraph.cpu()
            packs.append(awq.quantize(quant_config, target, target_inputs, device))

        last_subgraph = target.subgraph

    last_subgraph.cpu()

    LOGGER.info("Packing model...")
    awq.pack(quant_config, model, packs)

    LOGGER.info(f"Saving quantized model to {quant_name}")
    model.save_pretrained(quant_name)
    tokenizer.save_pretrained(quant_name)


if __name__ == "__main__":
    main()
