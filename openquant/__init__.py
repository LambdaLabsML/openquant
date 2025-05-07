from .subgraph import *

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
