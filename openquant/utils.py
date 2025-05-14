from contextlib import contextmanager
import gc
import logging
import psutil
import os


import torch


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
