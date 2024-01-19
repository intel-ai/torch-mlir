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
        "fuse-linalg-ops",
        "func.func(refback-generalize-tensor-pad)",
        "func.func(refback-generalize-tensor-concat)",
        # Apply some optimizations. It would be great if MLIR had more useful
        # optimizations that worked out of the box here.
        # Note: When measured, this doesn't seem to actually help that much
        # for the linalg-on-tensors backend.
        # This is likely because if things are naturally fusable we usually already
        # emit things in that form from the high level (e.g. single linalg-generic).
        # Other backends are likely to benefit more.
        "func.func(linalg-generalize-named-ops)",
        "func.func(linalg-fuse-elementwise-ops)",
        "convert-shape-to-std",
        # MLIR Sparsifier mini-pipeline. Note that this is the bare minimum
        # to ensure operations on sparse tensors are lowered to loops.
        "sparse-assembler",
        "sparsification-and-bufferization",
        "sparse-storage-specifier-to-llvm",
        "inline",  # inline sparse helper methods where useful
        # Bufferize.
        "func.func(scf-bufferize)",
        "func.func(tm-tensor-bufferize)",
        "func.func(empty-tensor-to-alloc-tensor)",
        "func.func(linalg-bufferize)",
        "func-bufferize",
        "arith-bufferize",
        "refback-mlprogram-bufferize",
        "func.func(tensor-bufferize)",
        "func.func(finalizing-bufferize)",
        "func.func(buffer-deallocation)",
        # Munge to make it ExecutionEngine compatible.
        # Specifically, we rewrite calling convention boundaries to be in terms
        # of unranked memref, and we rewrite the return to actually be a
        # callback that consumes the return (the final munged function always
        # returns void at the C level -- we get the return value by providing the
        # callback).
        "refback-munge-calling-conventions",
    ]
    if opts.use_kernels:
        # Introduce kernel calls for operations we want to execute using library
        # kernels.
        passes.append("convert-linalg-ops-to-kernel-calls")
    passes.extend(
        [
            # Insert global variable and instruction sequence for getting the next
            # global seed used in stateful rng.
            # Lower to LLVM
            "func.func(tm-tensor-to-loops)",
            "func.func(refback-munge-memref-copy)",
            "func.func(convert-linalg-to-loops)",
            "func.func(lower-affine)",
            "convert-scf-to-cf",
            "func.func(refback-expand-ops-for-llvm)",
            "func.func(arith-expand)",
            "func.func(convert-math-to-llvm)",
            # Handle some complex mlir::math ops (e.g. atan2)
            "convert-math-to-libm",
            "expand-strided-metadata",
            "finalize-memref-to-llvm",
            "lower-affine",
            "convert-bufferization-to-memref",
            "finalize-memref-to-llvm",
            "func.func(convert-arith-to-llvm)",
            "convert-func-to-llvm",
            "convert-cf-to-llvm",
            "convert-complex-to-llvm",
            "reconcile-unrealized-casts",
        ]
    )
    return "builtin.module(" + ",".join(passes) + ")"


class CpuProtoLinalgOnTensorsBackend(LinalgOnTensorsBackend):
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
        with DebugTimer(
            "CpuProtoLinalgOnTensorsBackend.compile()",
            logger=print if self._opts.debug_timer else None,
        ):
            run_pipeline_with_repro_report(
                imported_module,
                _build_lowering_pipeline(self._opts),
                "Lowering Linalg-on-Tensors IR to LLVM with RefBackend",
                enable_ir_printing=False,
                ir_dump_file=ir_file,
            )
        return imported_module

    def load(self, module) -> RefBackendInvoker:
        """Loads a compiled artifact into the runtime."""
        with DebugTimer(
            "CpuProtoLinalgOnTensorsBackend.load()",
            logger=print if self._opts.debug_timer else None,
        ):
            invoker = RefBackendInvoker(
                module,
                shared_libs=_collect_shared_libs(self._opts),
                logger=print if self._opts.debug_timer else None,
            )
        return invoker
