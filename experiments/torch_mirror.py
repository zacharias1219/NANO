"""Build a PyTorch model with exactly the same parameters as a Nano Sequential (oracle side only)."""
from __future__ import annotations

import numpy as np
import torch

from nano import nn


def mirror(model: nn.Sequential, dtype=torch.float64) -> torch.nn.Sequential:
    layers = []
    for m in model:
        if isinstance(m, nn.Linear):
            t = torch.nn.Linear(m.weight.shape[1], m.weight.shape[0], bias=m.bias is not None, dtype=dtype)
        elif isinstance(m, nn.Conv2d):
            oc, ic, k, _ = m.weight.shape
            t = torch.nn.Conv2d(ic, oc, k, stride=m.stride, padding=m.padding, dtype=dtype)
        elif isinstance(m, nn.MaxPool2d):
            t = torch.nn.MaxPool2d(m.k, m.stride)
        elif isinstance(m, nn.Flatten):
            t = torch.nn.Flatten()
        else:
            t = {nn.ReLU: torch.nn.ReLU, nn.Tanh: torch.nn.Tanh, nn.Sigmoid: torch.nn.Sigmoid}[type(m)]()
        if hasattr(m, "weight"):
            with torch.no_grad():
                t.weight.copy_(torch.from_numpy(np.array(m.weight.data)))  # same dtype: exact copy
                if getattr(m, "bias", None) is not None:
                    t.bias.copy_(torch.from_numpy(np.array(m.bias.data)))
        layers.append(t)
    return torch.nn.Sequential(*layers)


def torch_params(model: torch.nn.Module) -> list[np.ndarray]:
    """Parameters in Nano's order (weight then bias per layer)."""
    return [p.detach().numpy().copy() for p in model.parameters()]
