#ifndef GRAPH_PASSES
#define GRAPH_PASSES

include "mlir/Pass/PassBase.td"

def Tiling : Pass<"graph-tiling", "ModuleOp"> {
  let summary = "Tile shape calculations.";
  let constructor = [{
    mlir::graph::Graph::createTilingPass()
  }];
//   let options = [
//     Option<"extraLibrary", "extra-library", "std::string", /*default=*/"",
//            "MLIR module for splicing into the shape library">,
//   ];
  let description = [{
  }];
}

#endif // GRAPH_PASSES