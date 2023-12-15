set(MKL_LINK "static")
find_package(MKL CONFIG REQUIRED)

add_library(TorchMLIRKernels SHARED Kernels.cpp)

target_compile_options(TorchMLIRKernels PUBLIC $<TARGET_PROPERTY:MKL::MKL,INTERFACE_COMPILE_OPTIONS>)
target_include_directories(TorchMLIRKernels PUBLIC $<TARGET_PROPERTY:MKL::MKL,INTERFACE_INCLUDE_DIRECTORIES>)
target_link_libraries(TorchMLIRKernels PRIVATE $<LINK_ONLY:MKL::MKL>)

set_target_properties(TorchMLIRKernels PROPERTIES
         LIBRARY_OUTPUT_DIRECTORY "${TORCH_MLIR_PYTHON_PACKAGES_DIR}/torch_mlir/torch_mlir/_mlir_libs")

find_package(DNNL CONFIG REQUIRED)
add_library(TorchMLIRDNNKernels SHARED OneDNNKernels.cpp)
target_link_libraries(TorchMLIRDNNKernels PRIVATE DNNL::dnnl)

set_target_properties(TorchMLIRDNNKernels PROPERTIES
         LIBRARY_OUTPUT_DIRECTORY "${TORCH_MLIR_PYTHON_PACKAGES_DIR}/torch_mlir/torch_mlir/_mlir_libs")