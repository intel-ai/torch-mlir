import os

from torch_mlir_e2e_test.framework import TestOptions

def _find_shared_lib(name):
    this_file_dir = os.path.dirname(os.path.abspath(__file__))
    lib_file_path = f"{this_file_dir}/../../torch_mlir/_mlir_libs/{name}"
    if not os.path.isfile(lib_file_path):
        raise RuntimeError(f"Cannot find runtime library: {lib_file_path}")
    return lib_file_path


def _collect_shared_libs(opts: TestOptions):
    shared_libs = []
    if opts.use_kernels:
        shared_libs.append(_find_shared_lib("libTorchMLIRKernels.so"))
    if opts.use_gpu_runtime:
        shared_libs.append(_find_shared_lib("liblevel-zero-runtime.so"))
    return shared_libs

