"""A thin module layer (F5, F7): Linear, Conv2d, MaxPool2d, Flatten, activations, Sequential.

Modules hold Tensors with ``requires_grad=True`` and call ops; they add no
autodiff logic of their own. ``__call__`` pushes the module's name on a scope
stack so every tape node knows which layer produced it (used by the layerwise
gradcheck report), and runs forward hooks (used by probes).
"""
from __future__ import annotations

from collections.abc import Callable

import numpy as np

from . import init as winit
from . import tensor as _t
from .conv import conv2d, maxpool2d
from .ops import relu, sigmoid, tanh
from .tensor import Tensor


class Module:
    def __init__(self):
        self._hooks: list[Callable] = []
        self._name = type(self).__name__

    def forward(self, x):
        raise NotImplementedError

    def __call__(self, x):
        _t._SCOPE.append(self._name)
        try:
            out = self.forward(x)
        finally:
            _t._SCOPE.pop()
        for hook in self._hooks:
            hook(self, x, out)
        return out

    def register_forward_hook(self, fn: Callable) -> Callable:
        """``fn(module, input, output)`` runs after every forward; returns a remover."""
        self._hooks.append(fn)
        return lambda: self._hooks.remove(fn)

    def children(self) -> list[Module]:
        return [v for v in vars(self).values() if isinstance(v, Module)]

    def named_modules(self, prefix: str = ""):
        yield prefix or self._name, self
        for child in self.children():
            yield from child.named_modules(f"{prefix or self._name}/{child._name}")

    def named_parameters(self):
        for path, m in self.named_modules():
            for k, v in vars(m).items():
                if isinstance(v, Tensor) and v.requires_grad:
                    yield f"{path}.{k}", v

    def parameters(self) -> list[Tensor]:
        return [p for _, p in self.named_parameters()]

    def zero_grad(self) -> None:
        for p in self.parameters():
            p.grad = None

    def astype(self, dtype) -> Module:
        """Cast every parameter in place (float32 studies)."""
        for p in self.parameters():
            p.data = p.data.astype(dtype)
        return self

    def __repr__(self) -> str:
        return self._name


class Linear(Module):
    """y = x W^T + b with W of shape (out, in), the PyTorch layout. [D5.1]"""

    def __init__(self, n_in: int, n_out: int, init="he", bias: bool = True, rng=None):
        super().__init__()
        self.weight = Tensor(winit.get(init)((n_out, n_in), rng), requires_grad=True)
        self.bias = Tensor(np.zeros(n_out), requires_grad=True) if bias else None

    def forward(self, x):
        y = x @ self.weight.T
        return y + self.bias if self.bias is not None else y


class Conv2d(Module):
    def __init__(self, c_in: int, c_out: int, k: int, stride: int = 1, padding: int = 0, init="he", rng=None):
        super().__init__()
        self.weight = Tensor(winit.get(init)((c_out, c_in, k, k), rng), requires_grad=True)
        self.bias = Tensor(np.zeros(c_out), requires_grad=True)
        self.stride, self.padding = stride, padding

    def forward(self, x):
        return conv2d(x, self.weight, self.bias, stride=self.stride, padding=self.padding)


class MaxPool2d(Module):
    def __init__(self, k: int = 2, stride: int | None = None):
        super().__init__()
        self.k, self.stride = k, stride

    def forward(self, x):
        return maxpool2d(x, k=self.k, stride=self.stride)


class Flatten(Module):
    def forward(self, x):
        return x.flatten()


class ReLU(Module):
    def forward(self, x):
        return relu(x)


class Tanh(Module):
    def forward(self, x):
        return tanh(x)


class Sigmoid(Module):
    def forward(self, x):
        return sigmoid(x)


ACTIVATIONS = {"relu": ReLU, "tanh": Tanh, "sigmoid": Sigmoid}


class Sequential(Module):
    def __init__(self, *layers: Module):
        super().__init__()
        self.layers = list(layers)
        for i, layer in enumerate(self.layers):
            layer._name = f"{i}:{type(layer).__name__}"

    def children(self) -> list[Module]:
        return list(self.layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def __getitem__(self, i):
        return self.layers[i]

    def __len__(self) -> int:
        return len(self.layers)

    def __iter__(self):
        return iter(self.layers)


def mlp(n_in: int, width: int, depth: int, n_out: int, act: str = "relu", init="he", rng=None) -> Sequential:
    """``depth`` hidden layers of size ``width``; the output layer uses the same init."""
    layers, d = [], n_in
    for _ in range(depth):
        layers += [Linear(d, width, init=init, rng=rng), ACTIVATIONS[act]()]
        d = width
    return Sequential(*layers, Linear(d, n_out, init=init, rng=rng))
