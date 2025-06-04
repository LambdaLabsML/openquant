import argparse
import os
import logging

import torch
from transformers import AutoModel, AutoConfig, AutoModelForCausalLM

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
        quant_name += f"-W{args.weight_block_size}"

    logging.basicConfig(level=logging.INFO)

    LOGGER.info(args)

    weight_block_size = None
    if args.weight_block_size is not None:
        weight_block_size = [args.weight_block_size, args.weight_block_size]
    quant_config = fp8.QuantConfig(weight_block_size=weight_block_size)
    LOGGER.info(f"Using {quant_config}")

    config = AutoConfig.from_pretrained(args.model)
    if "ForConditionalGeneration" in config.architectures[0]:
        loader_cls = AutoModel
    else:
        assert "ForCausalLM"
        loader_cls = AutoModelForCausalLM

    LOGGER.info(f"Loading {args.model} with {loader_cls.__name__}")
    model = loader_cls.from_pretrained(args.model, torch_dtype="auto")

    plan = models.make_plan(model, include_experts=True)
    LOGGER.info(f"{len(plan)} quantization targets")

    LOGGER.info("Packing model...")
    fp8.pack(quant_config, model, plan)

    LOGGER.info(f"Saving quantized model to {quant_name}")
    model.save_pretrained(quant_name)
    utils.write_metadata(args, quant_name, model, torch.device("cpu"), 1)


if __name__ == "__main__":
    main()
