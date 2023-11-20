#include "torch-mlir/Conversion/FuseLinalg/FuseLinalg.h"

#include "../PassDetail.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MLProgram/IR/MLProgram.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include <algorithm>

namespace {

class MatmulTranspose : public mlir::OpRewritePattern<mlir::linalg::MatmulOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  mlir::LogicalResult
  matchAndRewrite(mlir::linalg::MatmulOp op,
                  mlir::PatternRewriter &rewriter) const override {

    auto mm_op = mlir::cast<mlir::linalg::MatmulOp>(&op);
    auto inputs = mm_op->getInputs();
    std::vector<int32_t> transposed_input_nums;
    transposed_input_nums.reserve(inputs.size());

    mlir::linalg::TransposeOp transposeOp = nullptr;
    std::fill_n(std::back_inserter(transposed_input_nums), inputs.size(), -1);
    if (transposed_input_nums.size() != 2) {
      return rewriter.notifyMatchFailure(
          op, "Inputs amount is not common for Matmul");
    }

    for (size_t num_input = 0; num_input < inputs.size(); num_input++) {
      mlir::Value input = inputs[num_input];
      transposeOp =
          llvm::dyn_cast<mlir::linalg::TransposeOp>(input.getDefiningOp());
      if (!transposeOp) {
        continue;
      }
      transposed_input_nums[num_input] = num_input;
    }

    const int default_inps = std::count(transposed_input_nums.cbegin(),
                                        transposed_input_nums.cend(), -1);
    if (default_inps != 1) {
      return rewriter.notifyMatchFailure(op, "Both Inputs is Transposed");
    }

    if (!transposeOp) {
      return rewriter.notifyMatchFailure(op, "Input is not TransposeOp");
    }
    /* Maybe we need this
    if (isListPotential[ERROR] lyMutated(transposeOp.getResult()))
      return rewriter.notifyMatchFailure(
          op, "TransposeOp result is potentially mutated");
    */

    mlir::Location loc = op.getLoc();

    mlir::SmallVector<mlir::Value> fusedInputOperands, fusedOutputOperands;
    mlir::SmallVector<mlir::Type> fusedResultTypes;
    for (mlir::OpOperand &opOperand : op.getOutputsMutable()) {
      fusedOutputOperands.push_back(opOperand.get());
      mlir::Type resultType = opOperand.get().getType();
      if (!mlir::isa<mlir::MemRefType>(resultType))
        fusedResultTypes.push_back(resultType);
    }

    mlir::Value matmul;
    if (transposed_input_nums[0] != -1) {
      fusedInputOperands.push_back(transposeOp.getInputMutable().get());
      fusedInputOperands.push_back(op.getInputsMutable()[1].get());

      auto mm_a = rewriter.create<mlir::linalg::MatmulTransposeAOp>(
          loc, fusedResultTypes, fusedInputOperands, fusedOutputOperands);
      matmul = mm_a.getResult(0);
    } else if (transposed_input_nums[1] != -1) {
      fusedInputOperands.push_back(op.getInputsMutable()[0].get());
      fusedInputOperands.push_back(transposeOp.getInputMutable().get());

      auto mm_b = rewriter.create<mlir::linalg::MatmulTransposeBOp>(
          loc, fusedResultTypes, fusedInputOperands, fusedOutputOperands);

      matmul = mm_b.getResult(0);
    }

    rewriter.replaceUsesWithIf(
        op.getResult(0), matmul, [&](mlir::OpOperand &use) {
          // Only replace consumer uses.
          return use.get().getDefiningOp() != transposeOp;
        });
    rewriter.eraseOp(op);

    if (transposeOp.getResult().use_empty()) {
      rewriter.eraseOp(transposeOp);
    } 
    return mlir::success();
  }
};

class FuseLinalgOps : public mlir::torch::FuseLinalgOpsBase<FuseLinalgOps> {
public:
  void runOnOperation() override {
    mlir::MLIRContext *context = &getContext();
    mlir::RewritePatternSet patterns(context);

    // pattern.add calls go here
    patterns.add<MatmulTranspose>(context);

    mlir::GreedyRewriteConfig config;
    config.useTopDownTraversal = true;

    if (failed(applyPatternsAndFoldGreedily(getOperation(), std::move(patterns),
                                            config))) {
      mlir::emitError(getOperation()->getLoc(), "failure in Linalg fusion");
      signalPassFailure();
    }
  }
};
} // namespace

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
mlir::torch::createFuseLinalgOpsPass() {
  return std::make_unique<FuseLinalgOps>();
}
