
#ifndef GRAPHDEMO_BASE
#define GRAPHDEMO_BASE

include "mlir/IR/OpBase.td"
include "TilingInterface.td"

def Graph_Dialect : Dialect {
  let name = "graph";
  let cppNamespace = "::mlir::graph::Graph";
  let description = [{
    ""
  }];
}


// class GraphOpTrait<string name> : NativeOpTrait<""> {
//   let trait = name;
//   let cppNamespace = "::mlir::graph::Graph::OpTrait";
// }

// def HasValueSemantics : GraphOpTrait<"HasValueSemantics">;
// def ReadOnly : GraphOpTrait<"ReadOnly">;
// def IsTrailingUnderscoreInplaceVariant
//   : GraphOpTrait<"IsTrailingUnderscoreInplaceVariant">;
// def AllowsTypeRefinement : GraphOpTrait<"AllowsTypeRefinement">;
// def AllowedInModuleInitializer : GraphOpTrait<"AllowedInModuleInitializer">;


// Base class for graph dialect operations. This operation inherits from the base
// `Op` class in OpBase.td, and provides:
//   * The parent dialect of the operation.
//   * The mnemonic for the operation, or the name without the dialect prefix.
//   * A list of traits for the operation.
class Graph_Op<string mnemonic, list<Trait> traits = []> :
    Op<Graph_Dialect, mnemonic, traits>;

//===----------------------------------------------------------------------===//
// MulOp
//===----------------------------------------------------------------------===//

def MulOp : Graph_Op<"mul", []> {
  let summary = "element-wise multiplication operation";
  let description = [{
    The "mul" operation performs element-wise multiplication between two
    tensors. The shapes of the tensor operands are expected to match.
  }];

  let arguments = (ins Arg<AnyTensor, "the input", []>:$lhs, 
                       Arg<AnyTensor, "the input", []>:$rhs);
  let results = (outs Res<AnyTensor, "", []>:$output);
}

def MatMulOp : Graph_Op<"matmul", [DeclareOpInterfaceMethods<TilingOpInterface>]> {
  let summary = "mat multiplication operation";
  let description = [{
    The "mul" operation performs mat multiplication between two
    tensors. The shapes of the tensor operands are expected to match.
  }];

  let arguments = (ins Arg<AnyTensor, "the input", []>:$lhs, 
                       Arg<AnyTensor, "the input", []>:$rhs);
  let results = (outs Res<AnyTensor, "", []>:$output);
}

//===----------------------------------------------------------------------===//
// MulOp
//===----------------------------------------------------------------------===//

def AddOp : Graph_Op<"add", []> {
  let summary = "element-wise add operation";
  let description = [{
    The "add" operation performs element-wise multiplication between two
    tensors. The shapes of the tensor operands are expected to match.
  }];

  let arguments = (ins Arg<AnyTensor, "the input", []>:$lhs, 
                       Arg<AnyTensor, "the input", []>:$rhs);
  let results = (outs Res<AnyTensor, "", []>:$output);
}

//===----------------------------------------------------------------------===//
// PrintOp
//===----------------------------------------------------------------------===//

def PrintOp : Graph_Op<"print", []> {
  let summary = "print operation";
  let description = [{
    The "print" builtin operation prints a given input tensor, and produces
    no results.
  }];

  // The print operation takes an input tensor to print.
  let arguments = (ins AnyTensor:$input);
}


#endif // GRAPHDEMO_BASE