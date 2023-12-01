//===----------------------------------------------------------------------===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#include "torch-mlir/InitAll.h"

#include "mlir/Dialect/Complex/IR/Complex.h"
#include "mlir/Dialect/Func/Extensions/InlinerExtension.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MLProgram/IR/MLProgram.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/IR/Dialect.h"
#include "torch-mlir-dialects/Dialect/TMTensor/IR/TMTensorDialect.h"
#include "torch-mlir-dialects/Dialect/TMTensor/Transforms/Passes.h"
#include "torch-mlir/Conversion/Passes.h"
#include "torch-mlir/Dialect/Torch/IR/TorchDialect.h"
#include "torch-mlir/Dialect/Torch/Transforms/Passes.h"
#include "torch-mlir/Dialect/TorchConversion/IR/TorchConversionDialect.h"
#include "torch-mlir/Dialect/TorchConversion/Transforms/Passes.h"
#include "torch-mlir/RefBackend/Passes.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVDialect.h"

#include "imex/Conversion/Passes.h"
#include "imex/InitIMEXDialects.h"
#include "imex/InitIMEXPasses.h"
#include "imex/Transforms/Passes.h"
#include "mlir/Dialect/Func/Extensions/AllExtensions.h"

void mlir::torch::registerAllDialects(mlir::DialectRegistry &registry) {
  registry.insert<mlir::func::FuncDialect>();
  registry.insert<mlir::torch::Torch::TorchDialect>();
  registry.insert<mlir::torch::TorchConversion::TorchConversionDialect>();
  registry.insert<mlir::torch::TMTensor::TMTensorDialect>();
  ::imex::registerAllDialects(registry);
  mlir::func::registerInlinerExtension(registry);
}

// TODO: Break this up when backends are separated.
void mlir::torch::registerOptionalInputDialects(
    mlir::DialectRegistry &registry) {
  registry.insert<complex::ComplexDialect, linalg::LinalgDialect,
                  memref::MemRefDialect, ml_program::MLProgramDialect,
                  scf::SCFDialect, tensor::TensorDialect, tosa::TosaDialect>();
  registry
      .insert<gpu::GPUDialect, spirv::SPIRVDialect, affine::AffineDialect>();
}

void mlir::torch::registerAllPasses() {
  mlir::torch::registerTorchPasses();
  mlir::torch::registerTorchConversionPasses();

  mlir::torch::registerConversionPasses();
  mlir::torch::TMTensor::registerPasses();

  ::imex::registerAllPasses();

#ifdef TORCH_MLIR_ENABLE_REFBACKEND
  mlir::torch::RefBackend::registerRefBackendPasses();
#endif
}
