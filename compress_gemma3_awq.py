import argparse
import os
import logging
import gc

import tqdm
import torch
import datasets
from transformers import AutoTokenizer, AutoModelForCausalLM, default_data_collator
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.modeling_utils import PreTrainedModel
from accelerate import cpu_offload
from accelerate.hooks import remove_hook_from_module

from openquant import (
    InputCatcher,
    awq,
    awq_transformers_quant_config,
    QuantConfig,
    ForwardPassEarlyStop,
    get_attn_implementation,
    AWQTarget,
    pack,
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
        default="google/gemma-3-27b-it",
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
    parser.add_argument("--disable-cpu-offload", default=False, action="store_true")
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

    quant_config = QuantConfig(num_bits=4)
    LOGGER.info(
        f"Using {quant_config}. Value range is: [{quant_config.min_int}, {quant_config.max_int}]"
    )

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

    LOGGER.info(f"Loading dataset {args.dataset}")
    ds = datasets.load_dataset(
        args.dataset, args.dataset_name, split=args.dataset_split
    )
    ds = ds.shuffle(seed=0).select(range(args.num_samples))
    ds = ds.map(preprocess)
    ds = ds.map(tokenize, remove_columns=ds.column_names)
    ds = ds.to_list()

    LOGGER.info(f"Loading {args.model}")
    attn_impl = get_attn_implementation()

    model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
        args.model, attn_implementation=attn_impl, torch_dtype="auto"
    )

    plan = make_awq_plan(model)
    num_targets = 0
    for _, targets in plan:
        for target in targets:
            LOGGER.info(f"{num_targets+1}. {target.names(model)}")
            num_targets += 1

    last_subgraph = None

    packing_targets = []

    pbar = tqdm.tqdm(total=num_targets)
    for subgraph, targets in plan:
        # compute new inputs for this subgraph
        if last_subgraph is None:
            # NOTE: gemma3 specific
            model.model.embed_tokens.to(device)
            model.model.rotary_emb.to(device)
            model.model.rotary_emb_local.to(device)

            catcher = InputCatcher(subgraph)
            for i in tqdm.tqdm(
                range(0, len(ds), args.batch_size),
                leave=False,
                desc="Capturing subgraph inputs",
            ):
                uncollated_batch = ds[i : i + args.batch_size]
                collated_batch = default_data_collator(uncollated_batch)
                model_inputs = model.prepare_inputs_for_generation(**collated_batch)
                model_inputs = {
                    k: v.to(device) if v is not None else v
                    for k, v in model_inputs.items()
                }
                try:
                    _ = model(**model_inputs)
                except ForwardPassEarlyStop:
                    pass
            subgraph_inputs = catcher.remove_handle_and_get()

            model.cpu()
        else:
            last_subgraph.to(device)
            for i in tqdm.tqdm(
                range(len(subgraph_inputs)),
                leave=False,
                desc="Capturing subgraph inputs",
            ):
                try:
                    subgraph_inputs[i][0] = last_subgraph(
                        *subgraph_inputs[i][0], **subgraph_inputs[i][1]
                    )
                except ForwardPassEarlyStop:
                    pass
            last_subgraph.cpu()

        assert len(subgraph_inputs) == len(ds) // args.batch_size, len(subgraph_inputs)

        # quantize each of the targets in this subgraph
        for target in targets:
            gc.collect()
            torch.cuda.empty_cache()
            LOGGER.debug(f"{torch.cuda.mem_get_info(device)[0] * 1e-9:.1f}GB available")

            # get inputs to target
            catcher = InputCatcher(target.scales)
            subgraph.to(device)
            for a, k in tqdm.tqdm(
                subgraph_inputs, leave=False, desc="Capturing layer inputs"
            ):
                try:
                    _ = subgraph(*a, **k)
                except ForwardPassEarlyStop:
                    pass
            target_inputs = catcher.remove_handle_and_get()

            # quantize it
            try:
                scales, zeros = awq(quant_config, target, target_inputs)
            except torch.OutOfMemoryError:
                LOGGER.debug("Sending subgraph back to CPU")
                subgraph.cpu()
                scales, zeros = awq(quant_config, target, target_inputs)

            for m, scale, zero in zip(target.scales, scales, zeros):
                packing_targets.append((m, scale, zero))

            pbar.update()

        last_subgraph = subgraph

    LOGGER.info("Packing model...")
    pack(quant_config, model, packing_targets)

    LOGGER.info(f"Saving quantized model to {quant_name}")
    model.config.quantization_config = awq_transformers_quant_config(quant_config)
    model.save_pretrained(quant_name)
    tokenizer.save_pretrained(quant_name)


from transformers.models.gemma3.modeling_gemma3 import (
    Gemma3ForCausalLM,
    Gemma3DecoderLayer,
)


def make_awq_plan(model) -> list[tuple[torch.nn.Module, list[AWQTarget]]]:
    assert isinstance(model, Gemma3ForCausalLM)

    plan = []
    decoder: Gemma3DecoderLayer
    for decoder in model.model.layers:
        subplan = [
            AWQTarget(
                inverse_scale=decoder.input_layernorm,
                scales=[
                    decoder.self_attn.q_proj,
                    decoder.self_attn.k_proj,
                    decoder.self_attn.v_proj,
                ],
            )
        ]
        if (
            decoder.self_attn.v_proj.out_features
            == decoder.self_attn.o_proj.in_features
        ):
            subplan.append(
                AWQTarget(
                    inverse_scale=decoder.self_attn.v_proj,
                    scales=[decoder.self_attn.o_proj],
                )
            )
        subplan.append(
            AWQTarget(
                inverse_scale=decoder.pre_feedforward_layernorm,
                scales=[
                    decoder.mlp.gate_proj,
                    decoder.mlp.up_proj,
                ],
            )
        )
        subplan.append(
            AWQTarget(
                inverse_scale=decoder.mlp.up_proj,
                scales=[decoder.mlp.down_proj],
            )
        )
        plan.append((decoder, subplan))
    return plan


if __name__ == "__main__":
    main()
