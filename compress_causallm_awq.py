import argparse
import os
import logging

import tqdm
import torch
import torch.distributed as dist
from torch.distributed.elastic.multiprocessing.errors import record
import datasets
from transformers import AutoTokenizer, AutoModelForCausalLM

from openquant import *

LOGGER = logging.getLogger(__name__)


@torch.inference_mode()
@record
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

    if "WORLD_SIZE" in os.environ:
        dist.init_process_group()
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        logging.basicConfig(
            format=f"[rank={rank}] [%(asctime)s] %(levelname)s:%(message)s",
            level=logging.INFO,
        )
    else:
        rank = 0
        world_size = 1
        logging.basicConfig(level=logging.INFO)

    LOGGER.info(args)
    LOGGER.info(os.environ)
    LOGGER.info(f"Using cuda:{rank}")
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)

    quant_config = awq.QuantConfig(num_bits=4, zero_point=not args.no_zero_point)
    LOGGER.info(
        f"Using {quant_config}. Value range is: [{quant_config.min_int}, {quant_config.max_int}]"
    )

    with rank0_first():
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

    with rank0_first():
        LOGGER.info(f"Loading dataset {args.dataset}")
        ds = datasets.load_dataset(
            args.dataset, args.dataset_name, split=args.dataset_split
        )
        ds = ds.shuffle(seed=0).select(range(args.num_samples))
        ds = ds.map(preprocess)
        ds = ds.map(tokenize, remove_columns=ds.column_names)
        ds = ds.to_list()

    attn_impl = get_attn_implementation()
    with rank0_first():
        LOGGER.info(f"Loading {args.model}")
        model = AutoModelForCausalLM.from_pretrained(
            args.model, attn_implementation=attn_impl, torch_dtype="auto"
        )

    plan = models.make_plan(model)
    for i, target in enumerate(plan):
        target_rank = i % world_size
        LOGGER.info(f"{i+1}. {target.names(model)} (on rank={target_rank})")

    cache_dir = f".cache/{quant_name}"
    with rank0_first():
        os.makedirs(cache_dir, exist_ok=True)

    last_subgraph = None
    for i, target in enumerate(plan):
        target_rank = i % world_size

        # compute new inputs for this subgraph
        if last_subgraph is None:
            models.head_to_device(model, device)
            subgraph_inputs = init_subgraph_inputs(
                model, target.subgraph, ds, args.batch_size, device
            )
            model.to("cpu", non_blocking=True)
        elif last_subgraph != target.subgraph:
            last_subgraph.to(device)
            update_subgraph_inputs(last_subgraph, subgraph_inputs)
            last_subgraph.to("cpu", non_blocking=True)

        last_subgraph = target.subgraph

        if target_rank != rank:
            continue

        if os.path.exists(f"{cache_dir}/{target.osname(model)}.pt"):
            LOGGER.info(f"Skipping {target.names(model)} (already exists in .cache/)")
            continue

        LOGGER.info(f"Quantizing {target.names(model)} on rank={target_rank}")
        target.subgraph.to(device)

        # get inputs to target
        target_inputs = get_layer_inputs(target.ops, target.subgraph, subgraph_inputs)

        # quantize it
        try:
            pack = awq.quantize(quant_config, target, target_inputs, device)
        except torch.OutOfMemoryError:
            LOGGER.error("Sending subgraph back to CPU - reduce batch size to avoid")
            target.subgraph.cpu()
            pack = awq.quantize(quant_config, target, target_inputs, device)

        torch.save(pack, f"{cache_dir}/{target.osname(model)}.pt")

    last_subgraph.to("cpu", non_blocking=True)

    if world_size > 1:
        LOGGER.info("Waiting for all ranks to finish quantizing")
        dist.barrier(device_ids=[rank])

    if rank == 0:
        LOGGER.info("Loading all packs")
        packs = []
        for target in tqdm.tqdm(plan, desc="Loading packs from cache", leave=False):
            packs.append(
                (target, *torch.load(f"{cache_dir}/{target.osname(model)}.pt"))
            )

        LOGGER.info("Packing model...")
        awq.pack(quant_config, model, packs)

        LOGGER.info(f"Saving quantized model to {quant_name}")
        model.save_pretrained(quant_name)
        tokenizer.save_pretrained(quant_name)

    if world_size > 1:
        LOGGER.info("Waiting for all ranks to finish before quitting.")
        dist.barrier(device_ids=[rank])
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
