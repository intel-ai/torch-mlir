# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Also available under a BSD-style license. See LICENSE.

import torch
import torch.nn as nn

from torch_mlir_e2e_test.framework import TestUtils
from torch_mlir_e2e_test.registry import register_test_case
from torch_mlir_e2e_test.annotations import annotate_args, export

# ==============================================================================

# Multi-layer perceptron (MLP) models.

class Mlp1LayerModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Reset seed to make model deterministic.
        torch.manual_seed(0)
        self.fc0 = nn.Linear(3, 5)
        self.tanh0 = nn.Tanh()
    @export
    @annotate_args([
        None,
        ([-1, -1], torch.float32, True),
    ])
    def forward(self, x):
        return self.tanh0(self.fc0(x))

@register_test_case(module_factory=lambda: Mlp1LayerModule())
def Mlp1LayerModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(5, 3))

class Mlp2LayerModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Reset seed to make model deterministic.
        torch.manual_seed(0)
        N_HIDDEN = 5
        self.fc0 = nn.Linear(3, N_HIDDEN)
        self.tanh0 = nn.Tanh()
        self.fc1 = nn.Linear(N_HIDDEN, 2)
        self.tanh1 = nn.Tanh()
    @export
    @annotate_args([
        None,
        ([-1, -1], torch.float32, True),
    ])
    def forward(self, x):
        x = self.tanh0(self.fc0(x))
        x = self.tanh1(self.fc1(x))
        return x

@register_test_case(module_factory=lambda: Mlp2LayerModule())
def Mlp2LayerModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(5, 3))

class Mlp2LayerModuleNoBias(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Reset seed to make model deterministic.
        torch.manual_seed(0)
        N_HIDDEN = 5
        self.fc0 = nn.Linear(3, N_HIDDEN, bias=False)
        self.tanh0 = nn.Tanh()
        self.fc1 = nn.Linear(N_HIDDEN, 2, bias=False)
        self.tanh1 = nn.Tanh()
    @export
    @annotate_args([
        None,
        ([-1, -1], torch.float32, True),
    ])
    def forward(self, x):
        x = self.tanh0(self.fc0(x))
        x = self.tanh1(self.fc1(x))
        return x

@register_test_case(module_factory=lambda: Mlp2LayerModuleNoBias())
def Mlp2LayerModuleNoBias_basic(module, tu: TestUtils):
    module.forward(tu.rand(5, 3))

class BatchMlpLayerModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Reset seed to make model deterministic.
        torch.manual_seed(0)
        self.fc0 = nn.Linear(3, 5)
        self.tanh0 = nn.Tanh()
    @export
    @annotate_args([
        None,
        ([-1, -1, -1], torch.float32, True),
    ])
    def forward(self, x):
        return self.tanh0(self.fc0(x))

@register_test_case(module_factory=lambda: BatchMlpLayerModule())
def BatchMlpLayerModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(7, 5, 3))


class MLP(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.flatten = torch.nn.Flatten()
        self.linear1 = torch.nn.Linear(input_dim, input_dim // 2)
        # self.relu = torch.nn.ReLU()
        # self.linear2 = torch.nn.Linear(input_dim // 2, output_dim)

    @export
    @annotate_args([
        None,
        ([-1, -1, -1], torch.float32, True),
    ])
    def forward(self, x):
        x = self.flatten(x)
        x = self.linear1(x)
        # x = self.relu(x)
        # x = self.linear2(x)
        return x

model = MLP(128 * 128, 0)

def model_factory():
    return model


test_input = torch.rand(1, 128, 128)
w = model.linear1.weight.detach().numpy()
b = model.linear1.bias.detach().numpy()
print("in shape: ", test_input.shape)
print(" w shape: ", w.shape)
print(" b shape: ", b.shape)


@register_test_case(module_factory=model_factory)
def MLP_basic(module, tu: TestUtils):
    out = module.forward(test_input)
    print("[test body] out shape: ", out.size())



from torchvision.models import (
    vgg16,
    resnet18,
    resnet50,
    resnext50_32x4d,
    resnext101_32x8d,
    densenet121,
    efficientnet_v2_m,
    mobilenet_v3_large,
)


def ResNext():
    torch.manual_seed(0)
    model = resnext50_32x4d()
    model.eval()
    return model


def ResNet():
    torch.manual_seed(0)
    model = resnet50()
    model.eval()
    return model


def Vgg():
    torch.manual_seed(0)
    model = vgg16()
    model.eval()
    return model


@register_test_case(module_factory=lambda: ResNext())
def ResNext_basic(module, tu: TestUtils):
    # out = module.forward(tu.randint(1, 11, high=13000))
    out = module.forward(tu.rand(1, 3, 224, 224))
    # model.forward(input_ids=input_ids.input_ids, attention_mask=input_ids.attention_mask, output_hidden_states=False, use_cache=False)
    # print("gen tokens: ", gen_tokens)
    return out


@register_test_case(module_factory=lambda: ResNet())
def ResNet_basic(module, tu: TestUtils):
    out = module.forward(tu.rand(1, 3, 224, 224))
    return out


@register_test_case(module_factory=lambda: Vgg())
def Vgg_basic(module, tu: TestUtils):
    out = module.forward(tu.rand(1, 3, 224, 224))
    return out
