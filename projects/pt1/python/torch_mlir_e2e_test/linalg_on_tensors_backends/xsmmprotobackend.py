# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Also available under a BSD-style license. See LICENSE.

import os

from torch_mlir.ir import *
from torch_mlir.passmanager import *
from torch_mlir.execution_engine import *
from torch_mlir.runtime import *
from torch_mlir.compiler_utils import run_pipeline_with_repro_report
from torch_mlir_e2e_test.framework import TestOptions, DebugTimer

from .abc import LinalgOnTensorsBackend
from .refbackend import RefBackendInvoker
from .utils import _collect_shared_libs

__all__ = [
    "CpuProtoLinalgOnTensorsBackend",
]


def _build_lowering_pipeline(opts: TestOptions):
    passes = [
        "linalg-generalize-named-ops",
        "default-tpp-passes",
        "refback-munge-calling-conventions",
        "finalize-memref-to-llvm",
        "expand-strided-metadata",
        "convert-tensor-to-linalg",
        "func.func(convert-linalg-to-loops)",
    ]
    if opts.use_omp:
        passes.append("convert-scf-to-openmp")
    passes.extend([
        "convert-vector-to-scf",
        "arith-expand",
        "lower-affine",
        "convert-vector-to-llvm",
        "finalize-memref-to-llvm",
        "convert-scf-to-cf"])
    if opts.use_omp:
        passes.append("convert-openmp-to-llvm")
    passes.extend([
        "convert-math-to-llvm",
        #"func.func(gpu-async-region)",
        #"gpu-to-llvm",
        #"gpu-module-to-binary{format=fatbin}",
        #"async-to-async-runtime",
        #"async-runtime-ref-counting",
        #"convert-async-to-llvm",
        "convert-func-to-llvm",
        "func.func(convert-arith-to-llvm)",
        "func.func(canonicalize)",
        "func.func(cse)",
        "reconcile-unrealized-casts",
        #"convert-gpu-launch-to-vulkan-launch",
        "symbol-dce",
    ])
    return "builtin.module(" + ",".join(passes) + ")"


class XsmmProtoLinalgOnTensorsBackend(LinalgOnTensorsBackend):
    """Main entry-point for the reference backend."""
    def __init__(self, opts: TestOptions = TestOptions()):
        super().__init__()
        self._opts = opts

    def compile(self, imported_module: Module, ir_file: str = None):
        """Compiles an imported module, with a flat list of functions.
        The module is expected to be in linalg-on-tensors + scalar code form.
        TODO: More clearly define the backend contract. Generally this will
        extend to support globals, lists, and other stuff.

        Args:
          imported_module: The MLIR module consisting of funcs in the torch
            dialect.
          ir_file: If specified, use it as output file for MLIR passes dumps
        Returns:
          An opaque, backend specific compiled artifact object that can be
          passed to `load`.
        """
        with DebugTimer('XsmmProtoLinalgOnTensorsBackend.compile()', logger=print if self._opts.debug_timer else None):
            run_pipeline_with_repro_report(
                imported_module, _build_lowering_pipeline(self._opts),
                "Lowering Linalg-on-Tensors IR to LLVM with XsmmBackend", ir_file)
        return imported_module

    def load(self, module) -> RefBackendInvoker:
        """Loads a compiled artifact into the runtime."""
        with DebugTimer('XsmmProtoLinalgOnTensorsBackend.load()', logger=print if self._opts.debug_timer else None):
            invoker = RefBackendInvoker(module,
                                        shared_libs=_collect_shared_libs(self._opts, ["libtpp_xsmm_runner_utils.so"]),
                                        logger=print if self._opts.debug_timer else None)
        return invoker
