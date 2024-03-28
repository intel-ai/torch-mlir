//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#include "torch-mlir/Dialect/GraphDEMO/IR/GraphDEMODialect.h"
#include "torch-mlir/Dialect/GraphDEMO/IR/GraphDEMOOps.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Transforms/InliningUtils.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::graph;
using namespace mlir::graph::Graph;

//===----------------------------------------------------------------------===//
// Dialect initialize method.
//===----------------------------------------------------------------------===//

void GraphDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "torch-mlir/Dialect/GraphDEMO/IR/GraphDEMOOps.cpp.inc"
      >();
  addTypes<
#define GET_TYPEDEF_LIST
#include "torch-mlir/Dialect/GraphDEMO/IR/GraphDEMOTypes.cpp.inc"
      >();
  // addInterfaces<TorchInlinerInterface>();
}

#include "torch-mlir/Dialect/GraphDEMO/IR/GraphDEMODialect.cpp.inc"
