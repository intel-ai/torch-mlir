# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Also available under a BSD-style license. See LICENSE.

from typing import List, Union, Optional, Sequence

import numpy as np
import torch
import torch.utils._pytree as pytree
from torch._dynamo.backends.common import aot_autograd
from torch._functorch.aot_autograd import (
    make_boxed_compiler,
    get_aot_compilation_context,
    set_model_name,
)

from torch_mlir._dynamo_fx_importer import import_fx_graph_as_func
from torch_mlir.dynamo import _get_decomposition_table
from torch_mlir.torchscript import (
    _example_args,
    OutputType,
    BACKEND_LEGAL_OPS,
    run_pipeline_with_repro_report,
    _lower_mlir_module,
    _canon_extra_library,
)
from torch_mlir_e2e_test.configs.utils import (
    recursively_convert_to_numpy,
    recursively_convert_from_numpy,
)
from torch_mlir_e2e_test.framework import TestConfig, TestOptions, Trace, TraceItem, DebugTimer

DUMPS_ENABLED = True

def _dump_repr_to_file(representation, filename: str):
    if not DUMPS_ENABLED:
        return

    with open(filename, 'w') as f:
        f.write(str(representation))

def refine_result_type(_result):
    if isinstance(_result, tuple):
        return tuple(refine_result_type(x) for x in _result)
    elif isinstance(_result, np.ndarray):
        return torch.from_numpy(_result)
    elif isinstance(_result, (bool, int, float)):
        return _result
    else:
        raise ValueError(f"Unhandled return type {type(_result)}")


def _returns_empty_tuple(fx_graph: torch.fx.GraphModule) -> bool:
    for node in fx_graph.graph.nodes:
        if node.op == "output":
            assert len(
                node.args) == 1, "Output node must have a single argument"
            node_arg = node.args[0]
            if node_arg != ():
                return False
    return True

# Replaces torch.aten.add.Tensor/torch.aten.mul.Tensor to 
# torch.aten.add.Scalar/torch.aten.mul.Scalar in case of Scalar argument
# Cannot be done on earlier stage, e.g. in _FXGraphImporter as it 
# needs to check argument types, which are not yet determined. 
# Maybe schema or target should be changed, but it decided in
# _dynamo eval_frame on pytorch side. Also Python schema not matches
# with mlir Schema - check include/torch-mlir/Dialect/Torch/IR/GeneratedTorchOps.td
# So in general it covers some of overload cases, which done on Python side automatically.
# e.g. conversion Scalar -> Tensor and vice versa
def scalarize_tensor_ops_on_scalars(gm: torch.fx.GraphModule):
    # Modify gm.graph
    for node in gm.graph.nodes:
        # Checks if we're calling a function (i.e:
        # torch.add)
        if node.op == 'call_function':
            # The target attribute is the function
            # that call_function calls.
            # call_function[target=torch.ops.aten.add.Tensor](args = (%arg64_1, 1), kwargs = {})
            if node.target == torch.ops.aten.add.Tensor:
                if len(node.args) != 2 or node.kwargs != {}:
                    continue
                elif not isinstance(node.args[1], torch.fx.node.Node):
                    node.target = torch.ops.aten.add.Scalar
            if node.target == torch.ops.aten.mul.Tensor:
                if len(node.args) != 2 or node.kwargs != {}:
                    continue
                elif not isinstance(node.args[1], torch.fx.node.Node):
                    node.target = torch.ops.aten.mul.Scalar

    gm.graph.lint() # Does some checks to make sure the

    # Recompile the forward() method of `gm` from its Graph
    gm.recompile()


def jit(
    model: torch.nn.Module,
    example_args: _example_args,
    symbol: str,
    opts: TestOptions,
    output_type: Union[str, "OutputType"] = OutputType.TORCH,
    backend_legal_ops: Optional[Sequence[str]] = None,
    extra_library=None,
    verbose: bool = False,
):
    if extra_library is None:
        extra_library = []
    import torch._dynamo as dynamo

    mlir_module = None

    extra_library_file_name = _canon_extra_library(extra_library)
    output_type = OutputType.get(output_type)
    if backend_legal_ops is not None:
        if output_type != OutputType.TORCH:
            raise Exception("`backend_legal_ops` is only valid with the "
                            "`torch` output type")
        backend_legal_ops = list(sorted(set(backend_legal_ops)))
    else:
        backend_legal_ops = BACKEND_LEGAL_OPS.get(output_type, [])

    @make_boxed_compiler
    def my_aot_autograd_backend(gm: torch.fx.GraphModule,
                                _example_inputs: List[torch.Tensor]):
        # Torch-MLIR does not support returning an empty tuple. The reason is
        # that both returning an empty tuple and returning `None` results in MLIR
        # functions that have as a return type `()`. In other words, there is no
        # way of differentiating between the two.
        assert not _returns_empty_tuple(gm), "encountered graph that does not return anything"

        scalarize_tensor_ops_on_scalars(gm)
        if opts.is_dump_enabled("fx-graph"):
            with open(f"{model._get_name()}.{symbol}-fx-graph.txt", "w") as f:
                print(gm.graph, file=f)

        nonlocal mlir_module
        *_, model_name, nth_graph = get_aot_compilation_context()
        mlir_module = import_fx_graph_as_func(gm.graph, model_name)

        if opts.is_dump_enabled("torch-mlir"):
            with open(f"{model._get_name()}.{symbol}-torch.mlir", "w") as f:
                print(mlir_module, file=f)

        return gm

    my_backend = aot_autograd(fw_compiler=my_aot_autograd_backend,
                              decompositions=_get_decomposition_table)

    with torch.no_grad():
        set_model_name(model.__class__.__name__)
        torch._dynamo.reset()
        dynamo_f = dynamo.optimize(my_backend, nopython=True)(
            lambda method, *inputs: method(*inputs))
        dynamo_f(lambda *inputs: model(*[x.clone() for x in inputs]),
                 *example_args)
        option_string = ("{backend-legal-ops=" + ",".join(backend_legal_ops) +
                         " extra-library=" + extra_library_file_name + "}")
        assert mlir_module is not None
        _dump_repr_to_file(mlir_module, 'forward.mlir')
        run_pipeline_with_repro_report(
            mlir_module,
            # f"builtin.module(torch-function-to-torch-backend-pipeline{option_string})",
            f"builtin.module(torch-lower-to-backend-contract)",
            "Lowering TorchFX IR -> Torch Backend IR",
        )

    ir_file = f"{model._get_name()}.{symbol}-torch-to-linanlg.txt" if opts.is_dump_enabled(
        "torch-mlir-lowering") else None
    return _lower_mlir_module(verbose, output_type, mlir_module, ir_file)


class TorchDynamoTestConfig(TestConfig):
    """TestConfig that runs the torch.nn.Module with TorchDynamo"""

    def __init__(self, backend, opts=TestOptions()):
        super().__init__()
        self.backend = backend
        self.opts = opts

    def compile(self, program: torch.nn.Module) -> torch.nn.Module:
        return program

    def run(self, artifact: torch.nn.Module, trace: Trace) -> Trace:
        result: Trace = []
        timing_logger = print if self.opts.is_debug_timer_enabled() else None
        with DebugTimer("TorchDynamoTestConfig.run()", logger=timing_logger):
            for item in trace:
                with DebugTimer("JIT", logger=timing_logger):
                    module = jit(artifact,
                                item.inputs,
                                item.symbol,
                                self.opts,
                                output_type="linalg-on-tensors")

                if self.opts.is_dump_enabled("linalg-mlir"):
                    with open(f"{artifact._get_name()}.{item.symbol}-linalg.mlir", "w") as f:
                        print(module, file=f)

                ir_file = f"{artifact._get_name()}.{item.symbol}-linalg-to-llvm.txt" if self.opts.is_dump_enabled(
                    "linalg-mlir-lowering") else None
                with DebugTimer("Backend.compile()", logger=timing_logger):
                    module = self.backend.compile(module, ir_file)

                if self.opts.is_dump_enabled("llvm-mlir"):
                    with open(f"{artifact._get_name()}.{item.symbol}-llvm.mlir", "w") as f:
                        print(module, file=f)

                with DebugTimer("Backend.load()", logger=timing_logger):
                    backend_module = self.backend.load(module)
                params = {
                    **dict(artifact.named_parameters(remove_duplicate=False)),
                    **dict(artifact.named_buffers(remove_duplicate=False)),
                }
                params_flat, params_spec = pytree.tree_flatten(params)
                params_flat = list(params_flat)
                with torch.no_grad():
                    with DebugTimer("recursively_convert_to_numpy", logger=timing_logger):
                        numpy_inputs = recursively_convert_to_numpy(params_flat +
                                                                    item.inputs)
                outputs = getattr(backend_module,
                                artifact.__class__.__name__)(*numpy_inputs)

                with DebugTimer("refine_result_type", logger=timing_logger):
                    output = refine_result_type(outputs)

                if self.opts.is_dump_enabled("obj"):
                    backend_module.ee.dump_to_object_file(f"{artifact._get_name()}.{item.symbol}.o")

                result.append(
                    TraceItem(symbol=item.symbol,
                            inputs=item.inputs,
                            output=output))
        return result
