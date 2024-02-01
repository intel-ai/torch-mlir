import os

from torch_mlir_e2e_test.framework import TestOptions

def _find_shared_lib(name):
    this_file_dir = os.path.dirname(os.path.abspath(__file__))

    search_dirs = [f"{this_file_dir}/../../torch_mlir/_mlir_libs",
                   f"{this_file_dir}/../../../../../../lib",
                   "/usr/lib"]
    if os.environ.get("CONDA_PREFIX", None):
        search_dirs.append(os.environ.get("CONDA_PREFIX") + "/lib")
    for dir in search_dirs:
        lib_file_path = f"{dir}/{name}"
        if os.path.isfile(lib_file_path):
            return lib_file_path

    raise RuntimeError(f"Cannot find runtime library {name}. Searched paths: {search_dirs}")


def _collect_shared_libs(opts: TestOptions, libs = []):
    shared_libs = [_find_shared_lib(lib) for lib in libs]
    if opts.use_kernels:
        shared_libs.append(_find_shared_lib("libTorchMLIRKernels.so"))
    if opts.use_omp:
        shared_libs.append(_find_shared_lib("libiomp5.so"))
    return shared_libs

