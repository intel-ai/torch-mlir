add_mlir_conversion_library(TorchMLIRTorchToGraph
  TorchToGraph.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/torch-mlir/Conversion/TorchToGraph

  DEPENDS
  TorchMLIRConversionPassIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  TorchMLIRGraphDialect
  TorchMLIRTorchDialect
  # TorchMLIRConversionUtils
)

torch_mlir_target_includes(TorchMLIRTorchToGraph)
