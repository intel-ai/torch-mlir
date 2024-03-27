//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v3.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-1.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#include "torch-mlir/Conversion/TorchToGraph/TorchToGraph.h"

#include "../PassDetail.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Traits.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Transforms/DialectConversion.h"
#include "torch-mlir/Conversion/Utils/Utils.h"
#include "torch-mlir/Dialect/GraphDEMO/IR/GraphDEMODialect.h"
#include "torch-mlir/Dialect/GraphDEMO/IR/GraphDEMOOps.h"
#include "torch-mlir/Dialect/Torch/IR/TorchDialect.h"
#include "torch-mlir/Dialect/Torch/IR/TorchOps.h"
#include "torch-mlir/Dialect/TorchConversion/IR/TorchConversionDialect.h"
#include "torch-mlir/Dialect/TorchConversion/IR/TorchConversionOps.h"
#include "torch-mlir/Dialect/TorchConversion/Transforms/BackendTypeConversion.h"

using namespace mlir;
using namespace mlir::torch;
using namespace mlir::torch::Torch;

namespace {

class ConvertAtenItemOp : public OpConversionPattern<AtenItemOp> {
public:
  using OpConversionPattern<AtenItemOp>::OpConversionPattern;
  using OpAdaptor = typename AtenItemOp::Adaptor;
  LogicalResult
  matchAndRewrite(AtenItemOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // auto operand = adaptor.getOperands()[0];
    // auto operandTy = cast<RankedTensorType>(operand.getType());
    // auto torchDTy =
    // cast<ValueTensorType>(op.getOperand().getType()).getDtype();

    // if (operandTy.getNumElements() != 1)
    //   return rewriter.notifyMatchFailure(op, "expected only one item");

    // auto zeroIdx = rewriter.create<arith::ConstantIndexOp>(op.getLoc(), 0);
    // auto rank = operandTy.getRank();
    // llvm::SmallVector<Value> indices(rank, zeroIdx);

    // Value extract = rewriter.create<tensor::ExtractOp>(
    //     op.getLoc(), operandTy.getElementType(), operand, indices);
    // auto extractTy = extract.getType();
    // if (isa<mlir::IntegerType>(extractTy) && !extractTy.isInteger(64)) {
    //   if (torchDTy.isUnsignedInteger()) {
    //     extract = rewriter.create<arith::ExtUIOp>(
    //         op.getLoc(), rewriter.getIntegerType(64), extract);
    //   } else {
    //     extract = rewriter.create<arith::ExtSIOp>(
    //         op.getLoc(), rewriter.getIntegerType(64), extract);
    //   }
    // }

    // if (isa<mlir::FloatType>(extractTy) && !extractTy.isF64()) {
    //   extract = rewriter.create<arith::ExtFOp>(op.getLoc(),
    //                                            rewriter.getF64Type(),
    //                                            extract);
    // }

    // rewriter.replaceOp(op, extract);
    return success();
  }
};

namespace {
class ConvertAtenAddmmOp : public OpConversionPattern<AtenAddmmOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(AtenAddmmOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    llvm::errs() << "Observing: " << op << "\n";
    Location loc = op->getLoc();
    Value mat1 = adaptor.getMat1();
    Value mat2 = adaptor.getMat2();

    auto mat1Type = mat1.getType().cast<RankedTensorType>();
    auto mat2Type = mat2.getType().cast<RankedTensorType>();

    // Get the rank of both matrix.
    unsigned mat1Rank = mat1Type.getRank();
    unsigned mat2Rank = mat2Type.getRank();

    Type newResultType = getTypeConverter()->convertType(op.getType());
    auto resultType = newResultType.cast<RankedTensorType>();
    Type elementType = resultType.getElementType();

    // The different cases of torch_matmul op is mentioned here:
    // https://pytorch.org/docs/stable/generated/torch.matmul.html

    // // First Case: Dot Product.
    // if (mat1Rank == 1 && mat2Rank == 1) {
    //   Value mat1Dim0 = getDimOp(rewriter, loc, mat1, 0);
    //   Value mat2Dim0 = getDimOp(rewriter, loc, mat2, 0);

    //   checkDimEqualHelper(rewriter, loc, mat1Dim0, mat2Dim0);

    //   Value zeroTensor = createZeroInitTensor(rewriter, loc, {},
    //   elementType); Value dotProd =
    //       rewriter
    //           .create<graph::Graph::MulOp>(loc, zeroTensor.getType(), mat1,
    //           mat2) .getResult();
    //   rewriter.replaceOpWithNewOp<tensor::CastOp>(op, newResultType,
    //   dotProd); return success();
    // }

    llvm::errs() << " ranks: " << mat1Rank << " mat2 rank: " << mat2Rank
                 << "\n";

    // Fourth Case: Vec-Vec Multiplication.
    if (mat1Rank == 2 && mat2Rank == 2) {
      Value mat1Dim0 = getDimOp(rewriter, loc, mat1, 0);
      Value mat1Dim1 = getDimOp(rewriter, loc, mat1, 1);
      Value mat2Dim0 = getDimOp(rewriter, loc, mat2, 0);
      Value mat2Dim1 = getDimOp(rewriter, loc, mat2, 1);
      // checkDimEqualHelper(rewriter, loc, mat1Dim1, mat2Dim0);

      // Value zeroTensor = createZeroInitTensor(
      //     rewriter, loc, ValueRange{mat1Dim0, mat2Dim1}, elementType);
      Value initTensor = rewriter.create<tensor::EmptyOp>(
          loc, getAsOpFoldResult(ValueRange{mat1Dim0, mat2Dim1}), elementType);
      Value matmul = rewriter
                         .create<graph::Graph::MatMulOp>(
                             loc, initTensor.getType(), mat1, mat2)
                         .getResult();
      // Value dotProd = rewriter
      //                     .create<graph::Graph::MulOp>(loc,
      //                     zeroTensor.getType(), mat1, mat2)
      //       .getResult();
      rewriter.replaceOpWithNewOp<tensor::CastOp>(op, newResultType, matmul);
      return success();
    }

    // Fifth Case: Batch-Matrix Multiplication.
    // TODO: Handle batch matrix multiplication when one of the matrix is unity
    // rank and the other has batch dimension.
    llvm::errs() << "Failed with ranks: " << mat1Rank
                 << " mat2 rank: " << mat2Rank << "\n";
    return failure();
  }
};
} // namespace

class ConvertTorchToGraph
    : public ConvertTorchToGraphBase<ConvertTorchToGraph> {
public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<graph::Graph::GraphDialect>();
    registry.insert<func::FuncDialect>();
    registry.insert<tensor::TensorDialect>();
    registry.insert<arith::ArithDialect>();
    TorchConversion::getBackendTypeConversionDependentDialects(registry);
  }
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ConversionTarget target(*context);
    target.addLegalDialect<graph::Graph::GraphDialect, func::FuncDialect,
                           tensor::TensorDialect, arith::ArithDialect>();
    target.addLegalDialect<graph::Graph::GraphDialect>();
    // target.addIllegalOp<Torch::AtenMatmulOp>();
    // target.addIllegalOp<Torch::Aten_ShapeAsTensorOp>();

    TypeConverter typeConverter;
    typeConverter.addConversion([](Type type) { return type; });
    TorchConversion::setupBackendTypeConversion(target, typeConverter);

    RewritePatternSet patterns(context);
    // patterns.add<ConvertAtenItemOp>(
    //     typeConverter, context);

    // target.addIllegalOp<AtenMmOp>();
    target.addIllegalOp<AtenAddmmOp>();
    patterns.add<ConvertAtenAddmmOp>(typeConverter, context);

    llvm::errs() << "op processing: " << getOperation() << "\n";
    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns))))
      return signalPassFailure();
  }
};

} // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::torch::createConvertTorchToGraphPass() {
  return std::make_unique<ConvertTorchToGraph>();
}
