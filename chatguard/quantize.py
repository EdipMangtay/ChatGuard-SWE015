"""Portable post-training, symmetric per-row INT8 weight quantization."""
from copy import deepcopy

import torch
from torch import nn
from torch.nn import functional as F


class Int8Matrix(nn.Module):
    def __init__(self, weight):
        super().__init__()
        scale = weight.detach().abs().amax(dim=1).clamp(min=1e-8) / 127
        codes = (weight.detach() / scale[:, None]).round().clamp(-127, 127).to(torch.int8)
        self.register_buffer("codes", codes)
        self.register_buffer("scale", scale)


class Int8Linear(Int8Matrix):
    def __init__(self, layer):
        super().__init__(layer.weight)
        bias = layer.bias.detach().clone() if layer.bias is not None else None
        self.register_buffer("bias", bias)

    def forward(self, x):
        return F.linear(x, self.codes.float() * self.scale[:, None], self.bias)


class Int8Embedding(Int8Matrix):
    def __init__(self, layer):
        super().__init__(layer.weight)

    def forward(self, ids):
        return self.codes[ids].float() * self.scale[ids].unsqueeze(-1)


def quantize_bert(model):
    model = deepcopy(model).eval()
    for name, module in list(model.named_modules()):
        if isinstance(module, nn.Linear):
            model.set_submodule(name, Int8Linear(module))
        elif isinstance(module, nn.Embedding):
            model.set_submodule(name, Int8Embedding(module))
    return model
