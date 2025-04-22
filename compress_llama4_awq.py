from transformers.models.llama4.modeling_llama4 import (
    Llama4ForConditionalGeneration,
    Llama4TextDecoderLayer,
    Llama4TextMoe,
    Llama4TextMLP,
    Llama4VisionEncoderLayer,
)


def awq_plan(model: Llama4ForConditionalGeneration):
    assert isinstance(model, Llama4ForConditionalGeneration)
    plan = []

    # encoder: Llama4VisionEncoderLayer
    # for encoder in model.vision_model.model.layers:
    #     plan.append(
    #         dict(
    #             inverse_scale=encoder.input_layernorm,
    #             scale=[
    #                 encoder.self_attn.q_proj,
    #                 encoder.self_attn.k_proj,
    #                 encoder.self_attn.v_proj,
    #             ],
    #         )
    #     )
    #     plan.append(
    #         dict(
    #             inverse_scale=encoder.self_attn.v_proj,
    #             scale=[encoder.self_attn.o_proj],
    #         )
    #     )
    #     plan.append(
    #         dict(
    #             inverse_scale=encoder.post_attention_layernorm,
    #             scale=[encoder.mlp.fc1],
    #         )
    #     )
    #     plan.append(
    #         dict(
    #             # TODO there is a GELU activation after fc1 & before fc2,
    #             # which is roughly x * Phi(x), which I think means we scale the output of fc1?
    #             inverse_scale=encoder.mlp.fc1,
    #             scale=[encoder.mlp.fc2],
    #         )
    #     )

    # text model
    decoder: Llama4TextDecoderLayer
    for decoder in model.language_model.model.layers:
        plan.append(
            dict(
                inverse_scale=decoder.input_layernorm,
                scale=[
                    decoder.self_attn.q_proj,
                    decoder.self_attn.k_proj,
                    decoder.self_attn.v_proj,
                ],
            )
        )
        plan.append(
            dict(
                inverse_scale=decoder.self_attn.v_proj,
                scale=[decoder.self_attn.o_proj],
            )
        )
        if decoder.is_moe_layer:
            assert isinstance(decoder.feed_forward, Llama4TextMoe)
            # NOTE: NOT quantizing experts yet
            plan.append(
                dict(
                    inverse_scale=decoder.post_attention_layernorm,
                    scale=[
                        decoder.feed_forward.shared_expert,
                        decoder.feed_forward.router,
                    ],
                )
            )
        else:
            assert isinstance(decoder.feed_forward, Llama4TextMLP)
            plan.append(
                dict(
                    inverse_scale=decoder.post_attention_layernorm,
                    scale=[
                        decoder.feed_forward.gate_proj,
                        decoder.feed_forward.up_proj,
                    ],
                )
            )

            plan.append(
                dict(
                    inverse_scale=decoder.feed_forward.up_proj,
                    scale=[decoder.feed_forward.down_proj],
                )
            )


import argparse
import os
import logging

import tqdm
import torch
import datasets
from transformers import AutoTokenizer, AutoModelForCausalLM, default_data_collator
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.modeling_utils import PreTrainedModel
from accelerate import cpu_offload

from openquant import (
    AWQ,
    QuantConfig,
    ForwardPassEarlyStop,
)

LOGGER = logging.getLogger(__name__)


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "-m",
        "--model",
        default="meta-llama/Llama-4-Scout-17B-16E-Instruct",
        help="The base model. Should be huggingface tag.",
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
    parser.add_argument("--cpu-offload", default=False, action="store_true")
    args = parser.parse_args()

    model_name = os.path.basename(args.model)
    quant_name = f"{model_name}-AWQ-Int4"

    logging.basicConfig(level=logging.INFO)

    LOGGER.info(args)
    LOGGER.info(os.environ)

    target_device = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]
    LOGGER.info(f"Using cuda:{target_device}")

    # NOTE: `0` index means the first **visible** cuda device
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(
        args.model, trust_remote_code=True
    )
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

    ds = datasets.load_dataset(
        args.dataset, args.dataset_name, split=args.dataset_split
    )
    ds = ds.shuffle(seed=0).select(range(args.num_samples))
    ds = ds.map(preprocess)
    ds = ds.map(tokenize, remove_columns=ds.column_names)
    ds = ds.to_list()

    try:
        import flash_attn

        attn_implementation = "flash_attention_2"
        LOGGER.info(f"Using flash attention")
    except ImportError:
        attn_implementation = None
        LOGGER.info(f"`import flash_attn` not found, pip install to use")

    if args.cpu_offload:
        model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
            args.model, attn_implementation=attn_implementation
        )
        cpu_offload(model, execution_device=device)
    else:
        with device:
            model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
                args.model, attn_implementation=attn_implementation
            )

    quant_config = QuantConfig(num_bits=4)

    LOGGER.info(
        f"Using {quant_config}. Value range is: [{quant_config.min_int}, {quant_config.max_int}]"
    )

    plan = awq_plan(model)

    pbar = tqdm.tqdm(total=len(plan))
    for target in plan.items():
        pbar.set_description(f"Quantizing {target}")

        quantizer = AWQ(
            quant_config,
            module_to_inverse_scale=target["inverse_scale"],
            modules_to_scale=target["scale"],
            execution_device=device,
            storage_device=torch.device("cpu"),
        )
        for m in quantizer.modules_to_scale:
            m.register_forward_pre_hook(quantizer.pre_forward_hook)

        # calibrate
        for i in tqdm.tqdm(
            range(0, len(ds), args.batch_size), leave=False, desc="Calibrating"
        ):
            uncollated_batch = ds[i : i + args.batch_size]
            collated_batch = default_data_collator(uncollated_batch)
            model_inputs = model.prepare_inputs_for_generation(**collated_batch)
            try:
                _ = model(**model_inputs)
            except ForwardPassEarlyStop:
                pass

        # do quantization
        quantized_module = quantizer.quantize_module()
        assert quantized_module is not None
        setattr(model, name, quantized_module)

        pbar.update()

    LOGGER.info(f"Saving quantized model to {quant_name}")
    model.config.quantization_config = AWQ.get_transformers_quant_config(quant_config)
    model.save_pretrained(quant_name)
    tokenizer.save_pretrained(quant_name)


if __name__ == "__main__":
    main()
