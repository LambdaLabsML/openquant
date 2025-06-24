import transformers
import transformers.models


def head_to_device(model, device):
    REGISTRY[type(model)].head(model, device)


def make_plan(model, *, include_experts: bool = False):
    return REGISTRY[type(model)].plan(model, include_experts=include_experts)


from .llama import Llama
from .llama4 import Llama4
from .mistral import Mistral
from .qwen3 import Qwen3
from .gemma3 import Gemma3

REGISTRY = {
    transformers.models.llama.LlamaForCausalLM: Llama,
    transformers.models.llama4.Llama4ForCausalLM: Llama4,
    transformers.models.llama4.Llama4ForConditionalGeneration: Llama4,
    transformers.models.gemma3.Gemma3ForCausalLM: Gemma3,
    transformers.models.gemma3.Gemma3ForConditionalGeneration: Gemma3,
    transformers.models.qwen3.Qwen3ForCausalLM: Qwen3,
    transformers.models.qwen3_moe.Qwen3MoeForCausalLM: Qwen3,
    transformers.models.mistral.MistralForCausalLM: Mistral,
}
