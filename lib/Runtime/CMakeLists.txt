add_library(TorchMLIRKernels SHARED Kernels.cpp)

set_target_properties(TorchMLIRKernels PROPERTIES
         LIBRARY_OUTPUT_DIRECTORY "${TORCH_MLIR_PYTHON_PACKAGES_DIR}/torch_mlir/torch_mlir/_mlir_libs")
