# Declare a dialect in the include directory
# function(add_mlir_dialect dialect dialect_namespace)
#   set(LLVM_TARGET_DEFINITIONS ${dialect}.td)
#   mlir_tablegen(${dialect}.h.inc -gen-op-decls)
#   mlir_tablegen(${dialect}.cpp.inc -gen-op-defs)
#   mlir_tablegen(${dialect}Types.h.inc -gen-typedef-decls -typedefs-dialect=${dialect_namespace})
#   mlir_tablegen(${dialect}Types.cpp.inc -gen-typedef-defs -typedefs-dialect=${dialect_namespace})
#   mlir_tablegen(${dialect}Dialect.h.inc -gen-dialect-decls -dialect=${dialect_namespace})
#   mlir_tablegen(${dialect}Dialect.cpp.inc -gen-dialect-defs -dialect=${dialect_namespace})
#   add_public_tablegen_target(MLIR${dialect}IncGen)
#   add_dependencies(mlir-headers MLIR${dialect}IncGen)
# endfunction()


set(LLVM_TARGET_DEFINITIONS GraphDEMOOps.td)
mlir_tablegen(GraphDEMOOps.h.inc -gen-op-decls)
mlir_tablegen(GraphDEMOOps.cpp.inc -gen-op-defs)

mlir_tablegen(GraphDEMOTypes.h.inc -gen-typedef-decls -typedefs-dialect=graph)
mlir_tablegen(GraphDEMOTypes.cpp.inc -gen-typedef-defs -typedefs-dialect=graph)

mlir_tablegen(GraphDEMODialect.h.inc -gen-dialect-decls -dialect=graph)
mlir_tablegen(GraphDEMODialect.cpp.inc -gen-dialect-defs -dialect=graph)
add_public_tablegen_target(MLIRGraphDEMOOpsIncGen)

set(LLVM_TARGET_DEFINITIONS TilingInterface.td)
mlir_tablegen(TilingInterface.h.inc -gen-op-interface-decls)
mlir_tablegen(TilingInterface.cpp.inc -gen-op-interface-defs)
add_public_tablegen_target(MLIRGraphDEMOInterfaceIncGen)

add_dependencies(mlir-headers MLIRGraphDEMOOpsIncGen MLIRGraphDEMOInterfaceIncGen)


# add_mlir_doc(Passes TorchMLIRTorchTransforms ./ -gen-pass-doc)

# set(LLVM_TARGET_DEFINITIONS TorchTypes.td)
# mlir_tablegen(TorchTypes.h.inc -gen-typedef-decls)
# mlir_tablegen(TorchTypes.cpp.inc -gen-typedef-defs)
# add_public_tablegen_target(MLIRTorchTypesIncGen)

# add_mlir_doc(TorchDialect TorchDialect Torch/ -gen-dialect-doc)
# add_mlir_doc(TorchOps TorchOps Torch/ -gen-op-doc)
