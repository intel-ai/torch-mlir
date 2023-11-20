#ifndef TORCHMLIR_FUSE_LINALG_H
#define TORCHMLIR_FUSE_LINALG_H

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir {
namespace torch {
std::unique_ptr<mlir::OperationPass<ModuleOp>> createFuseLinalgOpsPass();
}
} // namespace mlir

#endif // TORCHMLIR_FUSE_LINALG_H
