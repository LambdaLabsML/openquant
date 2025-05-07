import logging

import tqdm
import torch
import transformers
from transformers import default_data_collator


LOGGER = logging.getLogger(__name__)


class ForwardPassEarlyStop(Exception):
    pass


class QuantTarget:
    def __init__(self, *, parent: torch.nn.Module, ops: list[torch.nn.Module]):
        self.parent = parent
        self.ops = ops

    def names(self, root: torch.nn.Module):
        tmp = [""] * len(self.ops)
        for name, haystack in root.named_modules():
            for i in range(len(self.ops)):
                if haystack == self.ops[i]:
                    tmp[i] = name
                    break
        return tmp


class InputCatcher:
    def __init__(self, modules: list[torch.nn.Module]):
        self.modules = modules
        self.inputs = []
        self.handles = [
            m.register_forward_pre_hook(self.pre_forward_hook, with_kwargs=True)
            for m in self.modules
        ]

    def pre_forward_hook(self, module, args, kwargs):
        assert module in self.modules
        self.inputs.append([args_to(args, "cpu"), kwargs_to(kwargs, "cpu")])
        raise ForwardPassEarlyStop()

    def remove_handle_and_get(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        return self.inputs


def init_subgraph_inputs(
    model: "transformers.modeling_utils.PreTrainedModel",
    subgraph: torch.nn.Module,
    ds: list,
    batch_size: int,
    device: torch.device,
) -> list:
    catcher = InputCatcher([subgraph])
    for i in tqdm.tqdm(
        range(0, len(ds), batch_size),
        leave=False,
        desc="Capturing subgraph inputs",
    ):
        uncollated_batch = ds[i : i + batch_size]
        collated_batch = default_data_collator(uncollated_batch)
        model_inputs = model.prepare_inputs_for_generation(**collated_batch)
        model_inputs = {
            k: v.to(device) if v is not None else v for k, v in model_inputs.items()
        }
        try:
            _ = model(**model_inputs)
        except ForwardPassEarlyStop:
            pass
    return catcher.remove_handle_and_get()


def update_subgraph_inputs(subgraph: torch.nn.Module, subgraph_inputs: list):
    device = next(subgraph.parameters()).device

    for i in tqdm.tqdm(
        range(len(subgraph_inputs)),
        leave=False,
        desc="Capturing subgraph inputs",
    ):
        try:
            subgraph_inputs[i][0] = subgraph(
                *args_to(subgraph_inputs[i][0], device),
                **kwargs_to(subgraph_inputs[i][1], device),
            ).cpu()
        except ForwardPassEarlyStop:
            pass


def get_layer_inputs(
    layers: list[torch.nn.Module], subgraph: torch.nn.Module, subgraph_inputs: list
):
    device = next(subgraph.parameters()).device

    catcher = InputCatcher(layers)
    for a, k in tqdm.tqdm(subgraph_inputs, leave=False, desc="Capturing layer inputs"):
        try:
            _ = subgraph(*args_to(a, device), **kwargs_to(k, device))
        except ForwardPassEarlyStop:
            pass
    return catcher.remove_handle_and_get()


def args_to(args: list, device: torch.device) -> list:
    return [arg.to(device) if isinstance(arg, torch.Tensor) else arg for arg in args]


def kwargs_to(kwargs: list, device: torch.device) -> list:
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in kwargs.items()
    }
