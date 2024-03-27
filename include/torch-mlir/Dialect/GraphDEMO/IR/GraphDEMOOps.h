#ifndef GRAPH_DIALECT_GRAPHOPS_H
#define GRAPH_DIALECT_GRAPHOPS_H

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/CastInterfaces.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "torch-mlir/Dialect/GraphDEMO/IR/TilingInterface.h"

#define GET_OP_CLASSES
#include "torch-mlir/Dialect/GraphDEMO/IR/GraphDEMOOps.h.inc"

#endif // GRAPH_DIALECT_GRAPHOPS_H
