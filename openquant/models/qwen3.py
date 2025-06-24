from ..subgraph import QuantTarget


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
