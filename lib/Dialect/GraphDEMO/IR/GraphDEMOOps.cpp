//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//
#include "torch-mlir/Dialect/GraphDEMO/IR/GraphDEMOOps.h"
#include "llvm/Support/Debug.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/Casting.h"

using namespace mlir;
using namespace mlir::graph;
using namespace mlir::graph::Graph;

void mlir::graph::Graph::MatMulOp::tile() { 
    llvm::errs() << "in tile mm\n"; 
}

#define GET_OP_CLASSES
#include "torch-mlir/Dialect/GraphDEMO/IR/GraphDEMOOps.cpp.inc"