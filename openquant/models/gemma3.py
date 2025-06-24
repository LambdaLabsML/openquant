from ..subgraph import QuantTarget


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
