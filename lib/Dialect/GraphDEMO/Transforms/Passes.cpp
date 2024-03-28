#include "PassDetail.h"

#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "torch-mlir/Dialect/GraphDEMO/IR/GraphDEMODialect.h"
#include "torch-mlir/Dialect/GraphDEMO/IR/GraphDEMOOps.h"
#include "torch-mlir/Dialect/GraphDEMO/Transforms/Passes.h"

#include "torch-mlir/Dialect/GraphDEMO/IR/TilingInterface.cpp.inc"

using namespace mlir;
using namespace mlir::graph;
using namespace mlir::graph::Graph;

void mlir::graph::registerGraphPasses() {
  mlir::PassPipelineRegistration<>(
      "graph-pipeline", "Pipeline for graph demo.",
      [](OpPassManager &pm) { pm.addPass(createTilingPass()); });
}
namespace {
class TilingPass : public TilingBase<TilingPass> {
  void runOnOperation() override {
    auto module = getOperation();
    SmallVector<Operation *> toErase;

    llvm::errs() << "\n\nTiling base " << getOperation() << " body: \n\n\n";
    auto moduleBody = module.getBody(0);
    auto& func_op = moduleBody->getOperations().front();
    func_op.getBlock()->getOperations().front().dump();
    // moduleBody->getSuccessor(0)->getSuccessor(0)->dump();

    module.walk([&](mlir::Operation *op) {
      if (auto tileOp = dyn_cast<Tiling>(op)) {
        llvm::errs() << "Tiling shape for: " << *tileOp << "\n";
        tileOp.tile();
      } else {
        //   auto uses = op->getResult(0).getUses();
        bool haveUses = false;
        for (Value res : op->getResults()) {
          if (!res.getUses().empty()) {
            haveUses = true;
            break;
          }
        }
        llvm::errs() << "op " << *op << " have: " << haveUses << "\n";
      }
    });
  }

  void rewriteSignature(func::FuncOp func) {
    // Find the unique return op.
    func::ReturnOp returnOp;
    WalkResult walkResult = func.walk([&](func::ReturnOp op) {
      if (returnOp)
        return WalkResult::interrupt();
      returnOp = op;
      return WalkResult::advance();
    });
    if (walkResult.wasInterrupted()) {
      func.emitError() << "unimplemented: refining returns for function with "
                          "more than one return op";
      return signalPassFailure();
    }

    // Get the new operands. Either the original operand, or for tensors,
    // looking through TensorStaticInfoCastOp/CopyToNonValueTensorOp which are
    // presumed to have a more precise type.
    /*
    SmallVector<Value> newOperands;
    OpBuilder builder(returnOp);
    for (auto operand : returnOp.getOperands()) {
      Value newOperand = operand;
      // Look through TensorStaticInfoCastOp's, CopyToNonValueTensorOp's, and
      // DerefineOp's.
      for (;;) {
        if (auto cast = newOperand.getDefiningOp<TensorStaticInfoCastOp>()) {
          newOperand = cast.getOperand();
        } else if (auto copy =
                       newOperand.getDefiningOp<CopyToNonValueTensorOp>()) {
          // If the return (or transitively other ops) are not the only users,
          // then we can't be sure that the tensor hasn't been mutated, so stop
          // here.
          SetVector<Operation *> users(copy->getUsers().begin(),
                                       copy->getUsers().end());
          if (users.size() != 1)
            break;
          newOperand = copy.getOperand();
        } else if (auto derefine = newOperand.getDefiningOp<DerefineOp>()) {
          newOperand = derefine.getOperand();
        } else {
          break;
        }
      }

      if (auto tensorType = newOperand.getType().dyn_cast<BaseTensorType>()) {
        newOperands.push_back(
            copyTensorToType(builder, returnOp->getLoc(),
                             tensorType.getWithValueSemantics(), newOperand));
      } else {
        newOperands.push_back(newOperand);
      }
    }
    returnOp->setOperands(newOperands);

    // Update the function type.
    auto funcType = func.getFunctionType();
    func.setType(FunctionType::get(funcType.getContext(), funcType.getInputs(),
                                   ValueRange(newOperands).getTypes()));

    */
  }
};
} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
mlir::graph::Graph::createTilingPass() {
  return std::make_unique<TilingPass>();
}