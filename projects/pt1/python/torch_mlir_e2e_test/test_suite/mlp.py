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
    @annotate_args(
        [
            None,
            ([-1, -1], torch.float32, True),
        ]
    )
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
    @annotate_args(
        [
            None,
            ([-1, -1], torch.float32, True),
        ]
    )
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
    @annotate_args(
        [
            None,
            ([-1, -1], torch.float32, True),
        ]
    )
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
    @annotate_args(
        [
            None,
            ([-1, -1, -1], torch.float32, True),
        ]
    )
    def forward(self, x):
        return self.tanh0(self.fc0(x))


@register_test_case(module_factory=lambda: BatchMlpLayerModule())
def BatchMlpLayerModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(7, 5, 3))


class MLP(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.flatten = torch.nn.Flatten()
        self.linear1 = torch.nn.Linear(input_dim, input_dim // 2, bias=False)
        self.relu = torch.nn.ReLU()
        # self.linear2 = torch.nn.Linear(input_dim // 2, output_dim)

    @export
    @annotate_args([
        None,
        ([-1, -1, -1], torch.float32, True),
    ])
    def forward(self, x):
        x = self.flatten(x)
        x = self.linear1(x)
        x = self.relu(x)
        # x = self.linear2(x)
        return x

model = MLP(128 * 128, 1024)

with torch.no_grad():
    model.linear1.weight = nn.Parameter(torch.ones(model.linear1.weight.shape))
    model.linear1.weight[:, 1] = 2.
    model.linear1.weight[:, 4] = 5.
    model.linear1.weight[:, 6] = 5.
    # model.linear1.bias == nn.Parameter(torch.ones_like(model.linear1.bias))

def model_factory():
    return model

in_shape = (128, 128)

from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
import numpy as np


class RandomClsDataset(Dataset):
    def __init__(self, n, in_shape, n_classes):
        super().__init__()

        self.values = np.random.randn(n, *in_shape).astype(np.float32)
        self.labels = np.random.randint(n_classes, size=(n,))

    def __len__(self):
        return len(self.values)

    def __getitem__(self, index):
        return self.values[index], self.labels[index]


ds_size = 100
ds = RandomClsDataset(ds_size, in_shape, 100)
train_loader = DataLoader(
    ds, batch_size=100, shuffle=True, num_workers=1, pin_memory=False
)

from torch_mlir_e2e_test.framework import DebugTimer
print("Dataset size: ", len(train_loader.dataset))
sample_input = next(iter(train_loader))[0]
sample_input2 = next(iter(train_loader))[0]
sample_input3 = next(iter(train_loader))[0]
print("[in] sample")
with DebugTimer("\nVanilla sample", logger=print):
    model.forward(sample_input)
    # model.forward(sample_input2)
    # model.forward(sample_input3)
for _ in range(ds_size//100):
    with DebugTimer("\n**Inference** Vanilla", logger=print):
        out_vanilla = model.forward(sample_input2)


w = model.linear1.weight.detach().numpy()
# b = model.linear1.bias.detach().numpy()
print("in shape: ", sample_input.shape)
print(" w shape: ", w.shape)
# print(" b shape: ", b.shape)


@register_test_case(module_factory=model_factory)
def MLP_basic(module, tu: TestUtils):
    print("[in] sample")
    module.forward(sample_input)
    for _ in range(ds_size//100):
        out = module.forward(sample_input2)
    print("[test body] out shape: ", out.size())
