#ifndef TORCHMLIR_LINALG_TO_VECTOR_H
#define TORCHMLIR_LINALG_TO_VECTOR_H

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir {
namespace torch {
std::unique_ptr<mlir::OperationPass<ModuleOp>> createLinalgGenericToVectorOpsPass();
}
} // namespace mlir

#endif // TORCHMLIR_LINALG_TO_VECTOR_H
