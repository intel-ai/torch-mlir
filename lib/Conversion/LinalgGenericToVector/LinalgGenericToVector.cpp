
#include "torch-mlir/Conversion/LinalgGenericToVector/LinalgGenericToVector.h"

#include "../PassDetail.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/MLProgram/IR/MLProgram.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include <algorithm>

namespace {

class GenericVectorize
    : public mlir::OpRewritePattern<mlir::linalg::GenericOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  mlir::LogicalResult
  matchAndRewrite(mlir::linalg::GenericOp op,
                  mlir::PatternRewriter &rewriter) const override {

    return vectorize(rewriter, op, /*inputVectorSizes=*/{},
                     /*scalableVecDims=*/{}, false);
  }
};
class LinalgGenericToVectorOps
    : public mlir::torch::LinalgGenericToVectorOpsBase<
          LinalgGenericToVectorOps> {
public:
  void runOnOperation() override {
    mlir::MLIRContext *context = &getContext();
    mlir::RewritePatternSet patterns(context);

    patterns.add<GenericVectorize>(context);

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
mlir::torch::createLinalgGenericToVectorOpsPass() {
  return std::make_unique<LinalgGenericToVectorOps>();
}
