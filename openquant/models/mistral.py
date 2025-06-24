from ..subgraph import QuantTarget


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
