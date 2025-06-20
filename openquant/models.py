import transformers
import transformers.models

from .subgraph import QuantTarget


def head_to_device(model, device):
    REGISTRY[type(model)].head(model, device)


def make_plan(model, *, include_experts: bool = False) -> list[QuantTarget]:
    return REGISTRY[type(model)].plan(model, include_experts=include_experts)


class Llama:
    @staticmethod
    def head(model, device):
        model.model.embed_tokens.to(device=device)
        model.model.rotary_emb.to(device=device)

    @staticmethod
    def plan(model, include_experts: bool) -> list[QuantTarget]:
        from transformers.models.llama.modeling_llama import (
            LlamaDecoderLayer,
            LlamaForCausalLM,
        )

        assert isinstance(model, LlamaForCausalLM)

        plan = []
        decoder: LlamaDecoderLayer
        for decoder in model.model.layers:
            plan.append(
                QuantTarget(
                    subgraph=decoder,
                    parent=decoder.input_layernorm,
                    ops=[
                        decoder.self_attn.q_proj,
                        decoder.self_attn.k_proj,
                        decoder.self_attn.v_proj,
                    ],
                )
            )
            plan.append(
                QuantTarget(
                    subgraph=decoder,
                    parent=decoder.self_attn.v_proj,
                    ops=[decoder.self_attn.o_proj],
                )
            )
            plan.append(
                QuantTarget(
                    subgraph=decoder,
                    parent=decoder.post_attention_layernorm,
                    ops=[
                        decoder.mlp.gate_proj,
                        decoder.mlp.up_proj,
                    ],
                )
            )
            plan.append(
                QuantTarget(
                    subgraph=decoder,
                    parent=decoder.mlp.up_proj,
                    ops=[decoder.mlp.down_proj],
                )
            )
        return plan


class Gemma3:
    @staticmethod
    def head(model, device):
        model.model.embed_tokens.to(device=device)
        model.model.rotary_emb.to(device=device)
        model.model.rotary_emb_local.to(device=device)

    @staticmethod
    def plan(model, include_experts: bool) -> list[QuantTarget]:
        from transformers.models.gemma3.modeling_gemma3 import (
            Gemma3ForCausalLM,
            Gemma3DecoderLayer,
        )

        # TODO handle conditional generation with images
        assert isinstance(model, Gemma3ForCausalLM)

        plan = []
        decoder: Gemma3DecoderLayer
        for decoder in model.model.layers:
            plan.append(
                QuantTarget(
                    subgraph=decoder,
                    parent=decoder.input_layernorm,
                    ops=[
                        decoder.self_attn.q_proj,
                        decoder.self_attn.k_proj,
                        decoder.self_attn.v_proj,
                    ],
                )
            )
            plan.append(
                QuantTarget(
                    subgraph=decoder,
                    parent=decoder.self_attn.v_proj,
                    ops=[decoder.self_attn.o_proj],
                )
            )
            plan.append(
                QuantTarget(
                    subgraph=decoder,
                    parent=decoder.pre_feedforward_layernorm,
                    ops=[
                        decoder.mlp.gate_proj,
                        decoder.mlp.up_proj,
                    ],
                )
            )
            plan.append(
                QuantTarget(
                    subgraph=decoder,
                    parent=decoder.mlp.up_proj,
                    ops=[decoder.mlp.down_proj],
                )
            )
        return plan


class Llama4:
    @staticmethod
    def head(model, device):
        model.model.embed_tokens.to(device=device)
        model.model.rotary_emb.to(device=device)

    @staticmethod
    def plan(model, include_experts: bool) -> list[QuantTarget]:
        from transformers.models.llama4.modeling_llama4 import (
            Llama4ForConditionalGeneration,
            Llama4TextDecoderLayer,
            Llama4TextMoe,
            Llama4TextMLP,
            Llama4VisionEncoderLayer,
            Llama4ForCausalLM,
        )

        plan = []

        if isinstance(model, Llama4ForConditionalGeneration):
            # NOTE: VLLM doesn't support **ANY** quantized layers for vision model
            # https://github.com/vllm-project/vllm/blob/v0.9.0/vllm/model_executor/models/mllama4.py#L666
            # encoder: Llama4VisionEncoderLayer
            # for encoder in model.vision_model.model.layers:
            #     plan.extend(
            #         [
            #             QuantTarget(
            #                 subgraph=encoder,
            #                 parent=encoder.input_layernorm,
            #                 ops=[
            #                     encoder.self_attn.q_proj,
            #                     encoder.self_attn.k_proj,
            #                     encoder.self_attn.v_proj,
            #                 ],
            #             ),
            #             QuantTarget(
            #                 subgraph=encoder,
            #                 parent=encoder.self_attn.v_proj,
            #                 ops=[encoder.self_attn.o_proj],
            #             ),
            #             QuantTarget(
            #                 subgraph=encoder,
            #                 parent=encoder.post_attention_layernorm,
            #                 ops=[encoder.mlp.fc1],
            #             ),
            #             QuantTarget(
            #                 # TODO there is a GELU activation after fc1 & before fc2,
            #                 # which is roughly x * Phi(x), which I think means we scale the output of fc1?
            #                 subgraph=encoder,
            #                 parent=encoder.mlp.fc1,
            #                 ops=[encoder.mlp.fc2],
            #             ),
            #         ]
            #     )
            model = model.language_model

        assert isinstance(model, Llama4ForCausalLM)
        decoder: Llama4TextDecoderLayer
        for decoder in model.model.layers:
            plan.append(
                QuantTarget(
                    subgraph=decoder,
                    parent=decoder.input_layernorm,
                    ops=[
                        decoder.self_attn.q_proj,
                        decoder.self_attn.k_proj,
                        decoder.self_attn.v_proj,
                    ],
                )
            )
            plan.append(
                QuantTarget(
                    subgraph=decoder,
                    parent=decoder.self_attn.v_proj,
                    ops=[decoder.self_attn.o_proj],
                )
            )
            if decoder.is_moe_layer:
                assert isinstance(decoder.feed_forward, Llama4TextMoe)
                plan.append(
                    QuantTarget(
                        subgraph=decoder,
                        parent=decoder.post_attention_layernorm,
                        ops=[
                            decoder.feed_forward.shared_expert.gate_proj,
                            decoder.feed_forward.shared_expert.up_proj,
                            # NOTE: vllm does NOT support quantized router.
                            # https://github.com/vllm-project/vllm/blob/v0.9.0/vllm/model_executor/models/llama4.py#L70
                            # decoder.feed_forward.router,
                        ],
                    )
                )
                plan.append(
                    QuantTarget(
                        subgraph=decoder,
                        parent=decoder.feed_forward.shared_expert.up_proj,
                        ops=[decoder.feed_forward.shared_expert.down_proj],
                    )
                )
                if include_experts:
                    plan.append(
                        QuantTarget(
                            subgraph=decoder,
                            parent=decoder.post_attention_layernorm,
                            ops=[decoder.feed_forward.experts],
                        )
                    )
            else:
                assert isinstance(decoder.feed_forward, Llama4TextMLP)
                plan.append(
                    QuantTarget(
                        subgraph=decoder,
                        parent=decoder.post_attention_layernorm,
                        ops=[
                            decoder.feed_forward.gate_proj,
                            decoder.feed_forward.up_proj,
                        ],
                    )
                )
                plan.append(
                    QuantTarget(
                        subgraph=decoder,
                        parent=decoder.feed_forward.up_proj,
                        ops=[decoder.feed_forward.down_proj],
                    )
                )

        return plan


class Qwen3:
    @staticmethod
    def head(model, device):
        model.model.embed_tokens.to(device=device)
        model.model.rotary_emb.to(device=device)

    @staticmethod
    def plan(model, include_experts: bool) -> list[QuantTarget]:
        from transformers.models.qwen3.modeling_qwen3 import (
            Qwen3ForCausalLM,
            Qwen3DecoderLayer,
        )

        from transformers.models.qwen3_moe.modeling_qwen3_moe import (
            Qwen3MoeForCausalLM,
            Qwen3MoeDecoderLayer,
            Qwen3MoeMLP,
            Qwen3MoeSparseMoeBlock,
        )

        assert isinstance(model, (Qwen3ForCausalLM, Qwen3MoeForCausalLM))

        plan = []

        if isinstance(model, Qwen3ForCausalLM):
            decoder: Qwen3DecoderLayer
            for decoder in model.model.layers:
                plan.append(
                    QuantTarget(
                        subgraph=decoder,
                        parent=decoder.input_layernorm,
                        ops=[
                            decoder.self_attn.q_proj,
                            decoder.self_attn.k_proj,
                            decoder.self_attn.v_proj,
                        ],
                    )
                )
                plan.append(
                    QuantTarget(
                        subgraph=decoder,
                        parent=decoder.self_attn.v_proj,
                        ops=[decoder.self_attn.o_proj],
                    )
                )
                plan.append(
                    QuantTarget(
                        subgraph=decoder,
                        parent=decoder.post_attention_layernorm,
                        ops=[
                            decoder.mlp.gate_proj,
                            decoder.mlp.up_proj,
                        ],
                    )
                )
                plan.append(
                    QuantTarget(
                        subgraph=decoder,
                        parent=decoder.mlp.up_proj,
                        ops=[decoder.mlp.down_proj],
                    )
                )

        if isinstance(model, Qwen3MoeForCausalLM):
            decoder: Qwen3MoeDecoderLayer
            for decoder in model.model.layers:
                plan.append(
                    QuantTarget(
                        subgraph=decoder,
                        parent=decoder.input_layernorm,
                        ops=[
                            decoder.self_attn.q_proj,
                            decoder.self_attn.k_proj,
                            decoder.self_attn.v_proj,
                        ],
                    )
                )
                plan.append(
                    QuantTarget(
                        subgraph=decoder,
                        parent=decoder.self_attn.v_proj,
                        ops=[decoder.self_attn.o_proj],
                    )
                )

                if isinstance(decoder.mlp, Qwen3MoeMLP):
                    plan.append(
                        QuantTarget(
                            subgraph=decoder,
                            parent=decoder.post_attention_layernorm,
                            ops=[
                                decoder.mlp.gate_proj,
                                decoder.mlp.up_proj,
                            ],
                        )
                    )
                    plan.append(
                        QuantTarget(
                            subgraph=decoder,
                            parent=decoder.mlp.up_proj,
                            ops=[decoder.mlp.down_proj],
                        )
                    )
                elif isinstance(decoder.mlp, Qwen3MoeSparseMoeBlock):
                    # NOTE: vllm does NOT support quantized gate here
                    # plan.append(
                    #     QuantTarget(
                    #         subgraph=decoder,
                    #         parent=decoder.post_attention_layernorm,
                    #         ops=[decoder.mlp.gate],
                    #     )
                    # )
                    if include_experts:
                        plan.append(
                            QuantTarget(
                                subgraph=decoder,
                                parent=decoder.post_attention_layernorm,
                                ops=[decoder.mlp],
                            )
                        )
                else:
                    raise NotImplementedError(type(decoder.mlp))

        return plan


class Mistral:
    @staticmethod
    def head(model, device):
        model.model.embed_tokens.to(device=device)
        model.model.rotary_emb.to(device=device)

    @staticmethod
    def plan(model, include_experts: bool) -> list[QuantTarget]:
        from transformers.models.mistral.modeling_mistral import (
            MistralForCausalLM,
            MistralDecoderLayer,
        )

        assert isinstance(model, MistralForCausalLM)

        plan = []

        decoder: MistralDecoderLayer
        for decoder in model.model.layers:
            plan.append(
                QuantTarget(
                    subgraph=decoder,
                    parent=decoder.input_layernorm,
                    ops=[
                        decoder.self_attn.q_proj,
                        decoder.self_attn.k_proj,
                        decoder.self_attn.v_proj,
                    ],
                )
            )
            plan.append(
                QuantTarget(
                    subgraph=decoder,
                    parent=decoder.self_attn.v_proj,
                    ops=[decoder.self_attn.o_proj],
                )
            )
            plan.append(
                QuantTarget(
                    subgraph=decoder,
                    parent=decoder.post_attention_layernorm,
                    ops=[
                        decoder.mlp.gate_proj,
                        decoder.mlp.up_proj,
                    ],
                )
            )
            plan.append(
                QuantTarget(
                    subgraph=decoder,
                    parent=decoder.mlp.up_proj,
                    ops=[decoder.mlp.down_proj],
                )
            )

        return plan


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
