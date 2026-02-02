import torch
import torch.nn as nn
from collections import OrderedDict

def p_state_extract(model: nn.Module, x: torch.Tensor, p: int, 
                               desired: list[str] | None = None, 
                               keep_all: bool = False):
    """
    Run any model in p steps, collect intermediate states.

    Args:
        model: nn.Module (any model)
        x: input tensor
        p: number of steps to divide the model into
        desired: list of layer names to keep (optional)
        keep_all: if True, keep all intermediate activations (not just last in each step)

    Returns:
        states: dict with "input", "step_k", "output", and/or specific layers
    """
    # p = p-1 # Fixing index

    # --- get ordered list of submodules (skip root) ---
    layers = OrderedDict()
    for name, module in model.named_modules():
        if name != "":
            layers[name] = module

    layer_names = list(layers.keys())
    n = len(layer_names)
 
    chunk_size = (n + p - 1) // p   # ceil division
    # print('total '+str(n)+' layers detected - slice with size of '+str(chunk_size))
    # print(layer_names)

    states = {"input": x}
    outputs = {}

    # --- hook function ---
    def get_hook(name):
        def hook(_, __, output):
            outputs[name] = output.detach()
        return hook

    # register hooks
    handles = []
    for name, module in layers.items():
        handles.append(module.register_forward_hook(get_hook(name)))

    # run forward pass
    out = model(x)

    # remove hooks
    for h in handles:
        h.remove()

    # always store final output
    states["output"] = out

    # --- decide what to keep ---
    if desired is not None:  # explicit selection
        for name in desired:
            if name in outputs:
                states[name] = outputs[name]
    elif keep_all:  # keep everything
        states.update(outputs)
    else:  # default: just last in each chunk
        for step in range(p-1):
            start = step * chunk_size
            end   = min((step + 1) * chunk_size, n)
            if start >= end:
                continue
            last_name = layer_names[end - 1]
            states[f"step_{step+1}"] = outputs[last_name]

    return states
