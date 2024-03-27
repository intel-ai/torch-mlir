#ifndef TORCHMLIR_CONVERSION_TORCHTOGRAPH_TORCHTOGRAPH_H
#define TORCHMLIR_CONVERSION_TORCHTOGRAPH_TORCHTOGRAPH_H

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir {
namespace torch {
std::unique_ptr<OperationPass<func::FuncOp>> createConvertTorchToGraphPass();
} // namespace torch
} // namespace mlir

#endif // TORCHMLIR_CONVERSION_TORCHTOGRAPH_TORCHTOGRAPH_H
