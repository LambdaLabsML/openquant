import argparse
import os
import logging

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from openquant import *

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
    parser.add_argument("-w", "--weight-block-size", default=None, type=int)
    args = parser.parse_args()

    model_name = os.path.basename(args.model)
    quant_name = f"{model_name}-FP8"
    if args.weight_block_size is not None:
        quant_name = f"{model_name}-FP8-W{args.weight_block_size}"

    logging.basicConfig(level=logging.INFO)

    LOGGER.info(args)

    weight_block_size = None
    if args.weight_block_size is not None:
        weight_block_size = [args.weight_block_size, args.weight_block_size]
    quant_config = fp8.QuantConfig(weight_block_size=weight_block_size)
    LOGGER.info(f"Using {quant_config}")

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    LOGGER.info(f"Loading {args.model}")
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype="auto")

    plan = models.make_plan(model, include_experts=True)
    LOGGER.info(f"{len(plan)} quantization targets")

    LOGGER.info("Packing model...")
    fp8.pack(quant_config, model, plan)

    LOGGER.info(f"Saving quantized model to {quant_name}")
    model.save_pretrained(quant_name)
    tokenizer.save_pretrained(quant_name)
    utils.write_metadata(args, quant_name, torch.device("cpu"), 1)


if __name__ == "__main__":
    main()
