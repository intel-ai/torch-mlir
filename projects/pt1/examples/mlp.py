import torch
import torch_mlir
from torch_mlir_e2e_test.framework import TraceItem
from torch_mlir_e2e_test.configs.torchdynamo import TorchDynamoTestConfig
from torch_mlir_e2e_test.linalg_on_tensors_backends.refbackend import (
    RefBackendLinalgOnTensorsBackend,
)


class MLP(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.flatten = torch.nn.Flatten()
        self.linear1 = torch.nn.Linear(input_dim, input_dim // 2)
        self.relu = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(input_dim // 2, output_dim)

    def forward(self, x):
        x = self.flatten(x)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x


def model_factory():
    return MLP(16, 2)


torch.set_default_dtype(torch.float16)
model = model_factory()
test_input = torch.rand(2, 4, 4)

ref_res = [TraceItem(symbol="mlp", inputs=[test_input], output=model(test_input))]

config = TorchDynamoTestConfig(RefBackendLinalgOnTensorsBackend())
exp_res = config.run(model, ref_res)

print(ref_res[0].output)
print(exp_res[0].output)
