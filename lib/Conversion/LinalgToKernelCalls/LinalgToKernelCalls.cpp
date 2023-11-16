//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#include "torch-mlir/Conversion/LinalgToKernelCalls/LinalgToKernelCalls.h"

#include "../PassDetail.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Complex/IR/Complex.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "torch-mlir/Dialect/Torch/IR/TorchDialect.h"
#include "torch-mlir/Dialect/Torch/IR/TorchOps.h"
#include "torch-mlir/Dialect/TorchConversion/IR/TorchConversionDialect.h"
#include "torch-mlir/Dialect/TorchConversion/IR/TorchConversionOps.h"
#include "torch-mlir/Dialect/TorchConversion/Transforms/BackendTypeConversion.h"

#include <iostream>

using namespace mlir;
using namespace mlir::torch;
using namespace mlir::torch::Torch;

namespace {
LogicalResult
convertLinalgOpsInFunc(func::FuncOp func,
                       std::map<std::string, SmallVector<Type>> &usedKernels) {
  OpBuilder b(func.getBody());
  SmallVector<Operation *> replacedOps;
  func.walk([&](linalg::MatmulOp op) {
    auto types = op.getOperandTypes();
    auto lhs_type = types[0];
    auto rhs_type = types[1];
    auto res_type = types[2];
    auto lhs_elem_type = lhs_type.cast<BaseMemRefType>().getElementType();
    auto rhs_elem_type = rhs_type.cast<BaseMemRefType>().getElementType();
    auto res_elem_type = res_type.cast<BaseMemRefType>().getElementType();

    if (lhs_elem_type != rhs_elem_type || lhs_elem_type != res_elem_type)
      return;

    if (lhs_type.cast<BaseMemRefType>().getMemorySpace() !=
            rhs_type.cast<BaseMemRefType>().getMemorySpace() ||
        lhs_type.cast<BaseMemRefType>().getMemorySpace() !=
            res_type.cast<BaseMemRefType>().getMemorySpace())
      return;

    if (!lhs_elem_type.isF16() && !lhs_elem_type.isF32() &&
        !lhs_elem_type.isF64())
      return;

    b.setInsertionPoint(op);

    auto unranked_type = UnrankedMemRefType::get(
        lhs_elem_type, lhs_type.cast<BaseMemRefType>().getMemorySpace());
    std::string fn_name = "matmul_kernel_";
    llvm::raw_string_ostream rss(fn_name);
    lhs_elem_type.print(rss);
    if (!usedKernels.count(fn_name)) {
      usedKernels.emplace(
          fn_name,
          SmallVector<Type>({unranked_type, unranked_type, unranked_type}));
    }

    SmallVector<Value> unranked_ops;
    for (auto op : op.getOperands())
      unranked_ops.push_back(
          b.create<memref::CastOp>(op.getLoc(), unranked_type, op));
    b.create<func::CallOp>(op.getLoc(), fn_name, TypeRange({}), unranked_ops);
    replacedOps.push_back(op);
  });

  for (Operation *op : replacedOps)
    op->erase();

  return success();
}
} // namespace

namespace {
class ConvertLinalgOpsToKernelCalls
    : public ConvertLinalgOpsToKernelCallsBase<ConvertLinalgOpsToKernelCalls> {
public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect>();
    registry.insert<math::MathDialect>();
    registry.insert<func::FuncDialect>();
    registry.insert<tensor::TensorDialect>();
    registry.insert<arith::ArithDialect>();
    registry.insert<cf::ControlFlowDialect>();
    registry.insert<complex::ComplexDialect>();
  }

  void runOnOperation() override {
    auto module = getOperation();
    OpBuilder b(module.getBodyRegion());
    std::map<std::string, SmallVector<Type>> usedKernels;
    for (auto func : module.getOps<func::FuncOp>()) {
      if (failed(convertLinalgOpsInFunc(func, usedKernels)))
        return signalPassFailure();
    }

    // Create FuncOp for each used kernel function.
    for (auto &p : usedKernels) {
      auto kernelFunc = b.create<func::FuncOp>(
          module.getLoc(), p.first,
          FunctionType::get(module.getContext(), p.second, {}));
      kernelFunc.setPrivate();
    }
  }
};
} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
mlir::torch::createConvertLinalgOpsToKernelCallsPass() {
  return std::make_unique<ConvertLinalgOpsToKernelCalls>();
}
