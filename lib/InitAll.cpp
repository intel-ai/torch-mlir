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
#include "mlir/Dialect/Linalg/TransformOps/DialectExtension.h"
#include "mlir/Dialect/MLProgram/IR/MLProgram.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/TransformOps/TensorTransformOps.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/IR/Dialect.h"
#include "mlir/Target/LLVMIR/Dialect/All.h"
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

#include "TPP/Dialect/Check/BufferizableOpInterfaceImpl.h"
#include "TPP/Dialect/Check/CheckDialect.h"
#include "TPP/Dialect/Perf/BufferizableOpInterfaceImpl.h"
#include "TPP/Dialect/Perf/PerfDialect.h"
#include "TPP/Dialect/Tpp/BufferizableOpInterfaceImpl.h"
#include "TPP/Dialect/Tpp/TppDialect.h"
#include "TPP/Dialect/Transform/LinalgXTransformOps.h"
#include "TPP/Dialect/Xsmm/XsmmDialect.h"
#include "TPP/Passes.h"

void mlir::torch::registerAllDialects(mlir::DialectRegistry &registry) {
  registry.insert<mlir::func::FuncDialect>();
  registry.insert<mlir::torch::Torch::TorchDialect>();
  registry.insert<mlir::torch::TorchConversion::TorchConversionDialect>();
  registry.insert<mlir::torch::TMTensor::TMTensorDialect>();
  ::imex::registerAllDialects(registry);
  mlir::func::registerInlinerExtension(registry);

  registry.insert<mlir::tpp::TppDialect>();
  registry.insert<mlir::xsmm::XsmmDialect>();
  registry.insert<mlir::check::CheckDialect>();
  registry.insert<mlir::perf::PerfDialect>();
  mlir::linalgx::registerTransformDialectExtension(registry);
  mlir::check::registerBufferizableOpInterfaceExternalModels(registry);
  mlir::perf::registerBufferizableOpInterfaceExternalModels(registry);
  mlir::tpp::registerBufferizableOpInterfaceExternalModels(registry);

  // Add the following to include *all* MLIR Core dialects, or selectively
  // include what you need like above. You only need to register dialects that
  // will be *parsed* by the tool, not the one generated.
  registerAllDialects(registry);
  mlir::linalg::registerTransformDialectExtension(registry);
  mlir::tensor::registerTransformDialectExtension(registry);
  registerAllToLLVMIRTranslations(registry);
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
  mlir::tpp::registerTppCompilerPasses();
  mlir::tpp::registerTestStructuralMatchers();
  mlir::tpp::registerTestForToForAllRewrite();

#ifdef TORCH_MLIR_ENABLE_REFBACKEND
  mlir::torch::RefBackend::registerRefBackendPasses();
#endif
}
