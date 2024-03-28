set(MKL_LINK "static")
find_package(MKL CONFIG REQUIRED)

add_library(TorchMLIRKernels SHARED Kernels.cpp)

target_compile_options(TorchMLIRKernels PUBLIC $<TARGET_PROPERTY:MKL::MKL,INTERFACE_COMPILE_OPTIONS>)
target_include_directories(TorchMLIRKernels PUBLIC $<TARGET_PROPERTY:MKL::MKL,INTERFACE_INCLUDE_DIRECTORIES>)
target_link_libraries(TorchMLIRKernels PRIVATE $<LINK_ONLY:MKL::MKL>)

set_target_properties(TorchMLIRKernels PROPERTIES
         LIBRARY_OUTPUT_DIRECTORY "${TORCH_MLIR_PYTHON_PACKAGES_DIR}/torch_mlir/torch_mlir/_mlir_libs")


include (ExternalProject)

set(DNNL_SOURCE ${CMAKE_CURRENT_BINARY_DIR}/dnnl/src/dnnl/src)
set(DNNL_INSTALL ${CMAKE_CURRENT_BINARY_DIR}/dnnl/install)
set(DNNL_LIB_DIR ${DNNL_INSTALL}/${CMAKE_INSTALL_LIBDIR})
set(DNNL_INCLUDE_DIR ${DNNL_INSTALL}/include)
message (STATUS "ONE-DNN LIB dir: ${DNNL_LIB_DIR}")
message (STATUS "ONE-DNN Include dir: ${DNNL_INCLUDE_DIR}")

ExternalProject_Add(project_dnnl
    PREFIX dnnl
    GIT_REPOSITORY https://github.com/oneapi-src/onednn.git
    GIT_TAG 03691d7898b5a4fd1f471c8e30c97d394a5fdec2
    SOURCE_DIR ${DNNL_SOURCE}
    INSTALL_DIR ${DNNL_INSTALL}
    GIT_PROGRESS true
    USES_TERMINAL_BUILD true
    CMAKE_ARGS -DDNNL_BUILD_TESTS=OFF -DDNNL_ENABLE_CONCURRENT_EXEC=ON -DDNNL_BUILD_EXAMPLES=OFF -DDNNL_DEV_MODE=ON
    -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE} -DCMAKE_INSTALL_PREFIX=${DNNL_INSTALL}
)

include_directories(${DNNL_INCLUDE_DIR})
link_directories(${DNNL_LIB_DIR})

add_library(TorchMLIRDnnlKernels SHARED OneDnnKernels.cpp)

add_dependencies(TorchMLIRDnnlKernels project_dnnl)

include_directories(${DNNL_INCLUDE_DIR})
link_directories(${DNNL_LIB_DIR})

target_include_directories(TorchMLIRDnnlKernels PUBLIC ${DNNL_INCLUDE_DIR})
target_link_libraries(TorchMLIRDnnlKernels PRIVATE dnnl)

set_target_properties(TorchMLIRDnnlKernels PROPERTIES
         LIBRARY_OUTPUT_DIRECTORY "${TORCH_MLIR_PYTHON_PACKAGES_DIR}/torch_mlir/torch_mlir/_mlir_libs")

# ExternalProject_Add(ForexConnectDownload
#     PREFIX 3rd_party
#     #--Download step--------------
#     URL http://fxcodebase.com/bin/forexconnect/1.3.1/ForexConnectAPI-1.3.1-Linux-x86_64.tar.gz
#     URL_HASH SHA1=7fdb90a2d45085feb8b76167cae419ad4c211d6b
#     #--Configure step-------------
#     CONFIGURE_COMMAND ""
#     #--Build step-----------------
#     BUILD_COMMAND ""
#     #--Install step---------------
#     UPDATE_COMMAND "" # Skip annoying updates for every build
#     INSTALL_COMMAND ""
# )
