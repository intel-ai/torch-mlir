add_mlir_conversion_library(LinalgGenericToVector
  LinalgGenericToVector.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/torch-mlir/Conversion/LinalgGenericToVector

  DEPENDS
  TorchMLIRConversionPassIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  MLIRLinalgDialect
  MLIRVectorDialect
)

torch_mlir_target_includes(LinalgGenericToVector)
