from contextlib import contextmanager
import gc
import json
import logging
import shutil
import subprocess
import sys
import huggingface_hub
import psutil
import os


import torch
import yaml


LOGGER = logging.getLogger(__name__)


def get_attn_implementation():
    try:
        import flash_attn

        LOGGER.info(f"Using flash attention")
        return "flash_attention_2"
    except ImportError:
        LOGGER.warning(
            f"`import flash_attn` not found, run `pip install flash-attn --no-build-isolation`"
        )
        return None


def clean_memory(device: torch.device = None):
    gc.collect()
    torch.cuda.empty_cache()

    peak_gb = torch.cuda.max_memory_allocated(device) * 1e-9
    resv_gb = torch.cuda.memory_reserved(device) * 1e-9

    process = psutil.Process(os.getpid())
    cpu_mem_usage_gb = process.memory_info().rss * 1e-9
    LOGGER.debug(
        f"CPU {cpu_mem_usage_gb:.1f} GB | GPU {resv_gb:.1f} GB ({peak_gb:.1f} peak GB)"
    )


@contextmanager
def rank0_first():
    if torch.distributed.is_initialized():
        rank = torch.distributed.get_rank()
        if rank == 0:
            yield
        torch.distributed.barrier(device_ids=[rank])
        if rank > 0:
            yield
        torch.distributed.barrier(device_ids=[rank])
    else:
        yield


def set_submodule(
    root: torch.nn.Module, old: torch.nn.Module, new: torch.nn.Module
) -> str:
    module_name = None
    for name, haystack in root.named_modules():
        if haystack == old:
            module_name = name
            break
    assert module_name is not None

    # NOTE: we have to get the owner this way because torch stores list index as `.<index>.<next part>`,
    # and that is not compatible with getattr & lists
    owner = root
    name_parts = module_name.split(".")
    for attr in name_parts[:-1]:
        if attr.isnumeric():
            owner = owner[int(attr)]
        else:
            owner = getattr(owner, attr)
    setattr(owner, name_parts[-1], new)

    if isinstance(old, torch.nn.ModuleList):
        old_num_bytes = sum(
            [sum(p.numel() * p.dtype.itemsize for p in m.parameters()) for m in old]
        )
    else:
        old_num_bytes = sum(p.numel() * p.dtype.itemsize for p in old.parameters())
    new_num_bytes = sum(p.numel() * p.dtype.itemsize for p in new.parameters())
    LOGGER.info(
        f"Packed {module_name} from {old_num_bytes * 1e-6:.1f} mb to {new_num_bytes * 1e-6:.1f} mb"
    )

    return module_name


def write_metadata(args, metdata_dir, model, device: torch.device, world_size: int):
    os.makedirs(metdata_dir, exist_ok=True)

    LOGGER.info("Downloading base model readmes.")
    hf_cache_dir = huggingface_hub.snapshot_download(
        args.model, allow_patterns=["*.md", "*.json"]
    )
    for fname in os.listdir(hf_cache_dir):
        if fname == "model.safetensors.index.json":
            continue
        if fname.endswith("md") or fname.endswith("json"):
            dst = os.path.join(metdata_dir, fname)
            if os.path.exists(dst):
                LOGGER.info(f"Restoring {dst}")
            else:
                LOGGER.info(f"Copying {dst}")
            shutil.copy(os.path.join(hf_cache_dir, fname), dst)

    LOGGER.info("Adding quantization_config into config.json")
    with open(f"{metdata_dir}/config.json") as fp:
        config = json.load(fp)
    config["quantization_config"] = model.config.quantization_config
    with open(f"{metdata_dir}/config.json", "w") as fp:
        json.dump(config, fp, indent=2)

    try:
        commit_name = (
            subprocess.check_output(["git", "describe", "--exact-match", "--tags"])
            .decode()
            .strip()
        )
        LOGGER.info(f"tag {commit_name}")
    except subprocess.CalledProcessError:
        LOGGER.info("Unable to get current tag, using git tag instead")
        commit_name = (
            subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
        )
        LOGGER.info(f"commit {commit_name}")

    device_name = "1 CPUs"
    if device.type == "cuda":
        device_name = torch.cuda.get_device_name(device)
        device_name = f"{world_size}x`{device_name}` GPUs."

    python_command = "python"
    if world_size > 1:
        python_command = f"torchrun --nproc-per-node {world_size}"

    calibration_lines = []
    if hasattr(args, "dataset"):
        calibration_lines = [
            "",
            f"Calibrated with `{args.num_samples}` samples from `{args.dataset}`, `--batch-size {args.batch_size}`, `--seq-length {args.seq_length}`.",
        ]

    new_lines = [
        "# Quantization",
        f"Created with [openquant](https://github.com/LambdaLabsML/openquant/tree/{commit_name}) on `Python {sys.version}` with {device_name}",
        "",
        f"Base Model: [{args.model}](https://huggingface.co/{args.model})",
        *calibration_lines,
        "",
        "## Steps to reproduce:",
        f"1. `git clone https://github.com/LambdaLabsML/openquant`",
        f"2. `git checkout {commit_name}`",
        f"3. `{python_command} {' '.join(sys.argv)}`",
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
