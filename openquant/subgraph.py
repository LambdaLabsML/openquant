import logging

import tqdm
import torch
import transformers
from transformers import default_data_collator

from .utils import clean_memory


LOGGER = logging.getLogger(__name__)


class ForwardPassEarlyStop(Exception):
    pass


class QuantTarget:
    def __init__(
        self,
        *,
        subgraph: torch.nn.Module,
        parent: torch.nn.Module,
        ops: list[torch.nn.Module]
    ):
        self.subgraph = subgraph
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
        self.inputs.append([to_device(args, "cpu"), to_device(kwargs, "cpu")])
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
    clean_memory(device)

    catcher = InputCatcher([subgraph])
    for i in tqdm.tqdm(
        range(0, len(ds), batch_size),
        leave=False,
        desc="Initializing subgraph inputs",
    ):
        uncollated_batch = ds[i : i + batch_size]
        collated_batch = default_data_collator(uncollated_batch)
        model_inputs = model.prepare_inputs_for_generation(**collated_batch)
        x = {k: v.to(device) if v is not None else v for k, v in model_inputs.items()}
        x["use_cache"] = False
        try:
            model(**x)
        except ForwardPassEarlyStop:
            pass
    return catcher.remove_handle_and_get()


def update_subgraph_inputs(subgraph: torch.nn.Module, subgraph_inputs: list):
    device = next(subgraph.parameters()).device
    clean_memory(device)

    for i in tqdm.tqdm(
        range(len(subgraph_inputs)),
        leave=False,
        desc="Updating subgraph inputs",
    ):
        try:
            output = subgraph(
                *to_device(subgraph_inputs[i][0], device),
                **to_device(subgraph_inputs[i][1], device),
            )
            output = to_device(output, "cpu")
            if isinstance(output, torch.Tensor):
                output = (output,)
            subgraph_inputs[i][0] = output
        except ForwardPassEarlyStop:
            pass


def get_layer_inputs(
    layers: list[torch.nn.Module], subgraph: torch.nn.Module, subgraph_inputs: list
):
    device = next(subgraph.parameters()).device
    clean_memory(device)

    catcher = InputCatcher(layers)
    for a, k in tqdm.tqdm(subgraph_inputs, leave=False, desc="Capturing layer inputs"):
        try:
            _ = subgraph(*to_device(a, device), **to_device(k, device))
        except ForwardPassEarlyStop:
            pass
    return catcher.remove_handle_and_get()


def to_device(obj, device):
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    elif isinstance(obj, torch.Tensor):
        return obj.to(device)
    elif isinstance(obj, list):
        return [to_device(o, device) for o in obj]
    elif isinstance(obj, tuple):
        return tuple(to_device(o, device) for o in obj)
    elif isinstance(obj, dict):
        return {k: to_device(v, device) for k, v in obj.items()}
    else:
        raise NotImplementedError(type(obj))
