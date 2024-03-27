add_mlir_library(TorchMLIRGraphDialect
  GraphDEMODialect.cpp
  GraphDEMOOps.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/torch-mlir/Dialect/GraphDEMO

  DEPENDS
  MLIRGraphDEMOOpsIncGen
  MLIRGraphDEMOInterfaceIncGen
  # MLIRTorchTypesIncGen

  LINK_LIBS PUBLIC
  MLIRBytecodeOpInterface
  MLIRBytecodeReader
  MLIRBytecodeWriter
  MLIRFuncDialect
  MLIRIR
  MLIRSupport
  MLIRControlFlowInterfaces
  MLIRInferTypeOpInterface
  MLIRSideEffectInterfaces
)

torch_mlir_target_includes(TorchMLIRGraphDialect)
