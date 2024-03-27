//===- ShapeInferenceInterface.h - Interface definitions for ShapeInference -=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the declarations of the shape inference interfaces defined
// in ShapeInferenceInterface.td.
//
//===----------------------------------------------------------------------===//

#ifndef GRAPH_TILINGINTERFACE_H_
#define GRAPH_TILINGINTERFACE_H_

#include "mlir/IR/OpDefinition.h"

/// Include the auto-generated declarations.
#include "torch-mlir/Dialect/GraphDEMO/IR/TilingInterface.h.inc"


#endif // GRAPH_TILINGINTERFACE_H_
