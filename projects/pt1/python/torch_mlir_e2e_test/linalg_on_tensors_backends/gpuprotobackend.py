
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
    "GpuProtoLinalgOnTensorsBackend",
]

LOWERING_PIPELINE = "builtin.module(" + ",".join([
    "func.func(refback-generalize-tensor-pad)",
    # Apply some optimizations. It would be great if MLIR had more useful
    # optimizations that worked out of the box here.
    # Note: When measured, this doesn't seem to actually help that much
    # for the linalg-on-tensors backend.
    # This is likely because if things are naturally fusable we usually already
    # emit things in that form from the high level (e.g. single linalg-generic).
    # Other backends are likely to benefit more.
    "func.func(linalg-fuse-elementwise-ops)",
    "convert-shape-to-std",
    
    # # Bufferize (old).
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

    # Bufferize (new)
    # "arith-bufferize",
    # "func.func(tm-tensor-bufferize)",
    # "func.func(empty-tensor-to-alloc-tensor",
    #     "scf-bufferize",
    #     "shape-bufferize",
    #     "linalg-bufferize",
    #     "bufferization-bufferize",
    #     "tensor-bufferize)",
    # "func-bufferize",
    # "refback-mlprogram-bufferize", 

    
    # Insert global variable and instruction sequence for getting the next
    # global seed used in stateful rng.
    # Lower to LLVM
    "func.func(tm-tensor-to-loops)",
    "func.func(refback-munge-memref-copy)",

    # Lower to gpu
    # "func.func(finalizing-bufferize",
    # "func.func(",
    #       "convert-linalg-to-parallel-loops",
    #       "gpu-map-parallel-loops",
    #       "convert-parallel-loops-to-gpu)",
    "func.func(convert-linalg-to-parallel-loops)",
    "func.func(gpu-map-parallel-loops)",
    "func.func(convert-parallel-loops-to-gpu)",
    "func.func(insert-gpu-allocs{client-api=opencl})",
    "canonicalize",
    "normalize-memrefs",
    "func.func(lower-affine)",
    "gpu-kernel-outlining",
    "canonicalize",
    "cse",
    "set-spirv-capabilities{client-api=opencl}",
    "gpu.module(set-spirv-abi-attrs{client-api=opencl})",
    "canonicalize",
    "fold-memref-alias-ops",
    "imex-convert-gpu-to-spirv{enable-genisa-intrinsic}",
    "spirv.module(spirv-lower-abi-attrs",
             "spirv-update-vce)",
    "func.func(llvm-request-c-wrappers)",
    "serialize-spirv",
    "convert-gpu-to-gpux",

    # Munge to make it ExecutionEngine compatible.
    # Specifically, we rewrite calling convention boundaries to be in terms
    # of unranked memref, and we rewrite the return to actually be a
    # callback that consumes the return (the final munged function always
    # returns void at the C level -- we get the return value by providing the
    # callback).
    "refback-munge-calling-conventions",

    "convert-vector-to-scf",
    "convert-vector-to-scf",
    "convert-vector-to-llvm",
    "convert-index-to-llvm",
    "convert-arith-to-llvm",
    "convert-func-to-llvm",
    "convert-math-to-llvm",
    "convert-gpux-to-llvm",
    "convert-index-to-llvm",
    "expand-strided-metadata",
    "lower-affine",
    "finalize-memref-to-llvm",
    "reconcile-unrealized-casts",
]) + ")"


class GpuProtoLinalgOnTensorsBackend(LinalgOnTensorsBackend):
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
        with DebugTimer('GpuProtoLinalgOnTensorsBackend.compile()', logger=print if self._opts.debug_timer else None):
            run_pipeline_with_repro_report(
                imported_module, LOWERING_PIPELINE,
                "Lowering Linalg-on-Tensors IR to LLVM with RefBackend", ir_file)
        return imported_module

    def load(self, module) -> RefBackendInvoker:
        """Loads a compiled artifact into the runtime."""
        with DebugTimer('GpuProtoLinalgOnTensorsBackend.load()', logger=print if self._opts.debug_timer else None):
            invoker = RefBackendInvoker(module,
                                    shared_libs=_collect_shared_libs(self._opts))
        return invoker
