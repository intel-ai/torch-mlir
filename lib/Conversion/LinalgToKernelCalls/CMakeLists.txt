add_mlir_conversion_library(TorchMLIRLinalgToKernelCalls
  LinalgToKernelCalls.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/torch-mlir/Conversion/TorchToLinalg

  DEPENDS
  TorchMLIRConversionPassIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  MLIRLinalgDialect
  MLIRMathDialect
  TorchMLIRTorchDialect
)

torch_mlir_target_includes(TorchMLIRLinalgToKernelCalls)
