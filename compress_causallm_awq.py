import argparse
import json
import os
import logging
import shutil
import sys
import yaml
import subprocess

import huggingface_hub
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

    # Turns out the most reliable way to deallocate a module/tensor is to send it to the meta device
    # Even if you send the module to CPU and `del module` there's not really a great way to
    # force that to happen because of reference counting (it's extremely hard to get right)
    # Additionally, transformers by default will mmap the model memory when loaded, so the first
    # time you use the module it will be copied into RAM. If you **don't** deallocate it properly
    # you will basically end up with the entire model loaded into ram at the end.
    null_device = torch.device("meta")

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

    # NOTE: vllm doesn't have support for AWQ MoE layers, so we don't include experts
    plan = models.make_plan(model, include_experts=False)
    for target in plan:
        target.set_names(model)
    LOGGER.info(f"{len(plan)} quantization targets")

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
            models.head_to_device(model, null_device)
        elif last_subgraph != target.subgraph:
            last_subgraph.to(device)
            update_subgraph_inputs(last_subgraph, subgraph_inputs)
            last_subgraph.to(null_device)

        last_subgraph = target.subgraph

        if target_rank != rank:
            continue

        if os.path.exists(f"{cache_dir}/{target.osname}.pt"):
            LOGGER.info(f"Skipping {target.names} (already exists in .cache/)")
            continue

        LOGGER.info(f"{100 * (i / len(plan)):.0f}%: Quantizing {target.names}")
        target.subgraph.to(device)
        xs = get_layer_inputs(target.ops, target.subgraph, subgraph_inputs)

        # quantize it
        try:
            pack = awq.quantize(quant_config, target, xs, device)
        except torch.OutOfMemoryError:
            LOGGER.error("Sending subgraph back to CPU - reduce batch size to avoid")
            target.subgraph.cpu()
            pack = awq.quantize(quant_config, target, xs, device)

        torch.save(pack, f"{cache_dir}/{target.osname}.pt")

    last_subgraph.to(null_device)

    if world_size > 1:
        LOGGER.info("Waiting for all ranks to finish quantizing")
        dist.barrier(device_ids=[rank])

    if rank == 0:
        # NOTE: since we deleted model & all the targets earlier we need to reload them
        LOGGER.info("Reloading model")
        model = AutoModelForCausalLM.from_pretrained(
            args.model, attn_implementation=attn_impl, torch_dtype="auto"
        )

        LOGGER.info("Loading all packs")
        packs = []
        for target in tqdm.tqdm(plan, desc="Loading packs from cache", leave=False):
            packs.append((target, *torch.load(f"{cache_dir}/{target.osname}.pt")))

        LOGGER.info("Packing model...")
        awq.pack(quant_config, model, packs)

        LOGGER.info(f"Saving quantized model to {quant_name}")
        model.save_pretrained(quant_name)
        tokenizer.save_pretrained(quant_name)
        write_metadata(args, quant_name, device, world_size)

    if world_size > 1:
        LOGGER.info("Waiting for all ranks to finish before quitting.")
        dist.barrier(device_ids=[rank])
        dist.destroy_process_group()


def write_metadata(args, metdata_dir, device: torch.device, world_size: int):
    os.makedirs(metdata_dir, exist_ok=True)

    LOGGER.info("Downloading base model readmes.")
    hf_cache_dir = huggingface_hub.snapshot_download(args.model, allow_patterns="*.md")
    for fname in os.listdir(hf_cache_dir):
        if fname.endswith("md"):
            LOGGER.info(f"Copying {hf_cache_dir}/{fname} into {metdata_dir}")
            shutil.copy(
                os.path.join(hf_cache_dir, fname),
                os.path.join(metdata_dir, fname),
            )

    commit_hash = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
    LOGGER.info(f"commit {commit_hash}")

    assert device.type == "cuda"
    device_name = torch.cuda.get_device_name(device)

    python_command = "python"
    if world_size > 1:
        python_command = f"torchrun --nproc-per-node {world_size}"

    new_lines = [
        "# Quantization",
        f"Created with [openquant](https://github.com/LambdaLabsML/openquant/tree/{commit_hash}) on `Python {sys.version}` with {world_size}x`{device_name}` GPUs.",
        "",
        f"Base Model: [{args.model}](https://huggingface.co/{args.model})",
        "",
        f"Calibrated with `{args.num_samples}` samples from `{args.dataset}`, `--batch-size {args.batch_size}`, `--seq-length {args.seq_length}`.",
        "",
        "## Steps to reproduce:",
        f"1. `git clone https://github.com/LambdaLabsML/openquant`",
        f"2. `git checkout {commit_hash}`",
        f"3. `{python_command} {' '.join(sys.argv)}`",
        "",
        "## Evaluation",
        "TODO",
        "",
        "# Base Model README.md",
        "",
    ]

    with open(f"{metdata_dir}/README.md") as fp:
        readme_content = fp.read()

    metadata_start = readme_content.find("---")
    if metadata_start >= 0:
        metadata_end = readme_content.find("---", metadata_start + len("---")) + len(
            "---"
        )
        metadata = readme_content[metadata_start:metadata_end]
        readme_content = readme_content[:metadata_start] + readme_content[metadata_end:]
    else:
        metadata = "\n".join(
            [
                "---",
                f'base_model: "{args.model}"',
                "---",
            ]
        )

    metadata = yaml.safe_load(metadata.replace("---", ""))
    if "base_model" not in metadata:
        metadata["base_model"] = args.model
    if "license" not in metadata:
        metadata["license"] = "mit"
    metadata = "---\n" + yaml.dump(metadata) + "---\n"

    new_content = "\n".join(new_lines)
    LOGGER.info(f"Writing {new_content} into README.md")
    with open(f"{metdata_dir}/README.md", "w") as fp:
        fp.write(metadata + "\n" + new_content + "\n" + readme_content)

    LOGGER.info(f"Dumping `pip freeze` to {metdata_dir}/openquant-requirements.txt")
    freeze = subprocess.check_output(["pip", "freeze"]).decode()
    with open(f"{metdata_dir}/openquant-requirements.txt", "w") as fp:
        fp.write(freeze)

    LOGGER.info(f"Dumping `args` to {metdata_dir}/openquant-args.json")
    with open(f"{metdata_dir}/openquant-args.json", "w") as fp:
        json.dump(vars(args), fp)


if __name__ == "__main__":
    main()
