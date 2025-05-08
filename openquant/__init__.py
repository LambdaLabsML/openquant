from .subgraph import *

import gc
import logging


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
    peak_gb = torch.cuda.max_memory_allocated(device) * 1e-9
    torch.cuda.reset_peak_memory_stats(device)

    gc.collect()
    torch.cuda.empty_cache()
    free_gb = torch.cuda.mem_get_info(device)[0] * 1e-9
    resv_gb = torch.cuda.memory_reserved(device) * 1e-9
    LOGGER.info(
        f"{free_gb + resv_gb:.1f}GB available ({free_gb:.1f}GB free + {resv_gb:.1f}GB reserved) | Peak memory: {peak_gb:.1f}GB"
    )
