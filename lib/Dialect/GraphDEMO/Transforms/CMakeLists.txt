add_mlir_library(TorchMLIRGraphPasses
  Passes.cpp
  

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/torch-mlir/Dialect/GraphDEMO/Transforms

  DEPENDS
  MLIRGraphDEMOPassIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  MLIRTransforms
  TorchMLIRGraphDialect
)

torch_mlir_target_includes(TorchMLIRGraphPasses)
