#ifndef GRAPH_DIALECT_TRANSFORMS_PASSES_H
#define GRAPH_DIALECT_TRANSFORMS_PASSES_H

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"

#include <memory>

namespace mlir {
class ModuleOp;

namespace graph {
namespace Graph {

#include "torch-mlir/Dialect/GraphDEMO/Transforms/Passes.h.inc"

std::unique_ptr<OperationPass<ModuleOp>> createTilingPass();

} // namespace Graph


void registerGraphPasses();

//===----------------------------------------------------------------------===//
// Pass registration
//===----------------------------------------------------------------------===//

#define GEN_PASS_REGISTRATION
#include "torch-mlir/Dialect/GraphDEMO/Transforms/Passes.h.inc"

} // namespace graph
} // namespace mlir

#endif // TORCHMLIR_DIALECT_TORCH_TRANSFORMS_PASSES_H
