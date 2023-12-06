#!/bin/bash

set -ex
cd $(dirname "$0")/../

project_dir=$PWD
echo "Using project dir: ${project_dir}"

# assuming git submodule update --init was done and all the code is present
pushd externals/mlir-extensions
git checkout tags/v0.3
git apply ${project_dir}/utils/public-deps.patch
git apply ${project_dir}/utils/level-zero-runtime-log.patch
popd

pushd externals/llvm-project
git checkout `cat ${project_dir}/externals/mlir-extensions/build_tools/llvm_version.txt`
git apply ${project_dir}/utils/llvm-error-msg.patch
popd


cmake -GNinja -Bbuild \
    -DCMAKE_BUILD_TYPE=Release \
    -DPython3_FIND_VIRTUALENV=ONLY \
    -DLLVM_ENABLE_PROJECTS=mlir \
    -DLLVM_EXTERNAL_PROJECTS="torch-mlir;Imex" \
    -DLLVM_EXTERNAL_TORCH_MLIR_SOURCE_DIR="$PWD" \
    -DLLVM_EXTERNAL_IMEX_SOURCE_DIR="$PWD/externals/mlir-extensions" \
    -DMLIR_ENABLE_BINDINGS_PYTHON=ON \
    -DLLVM_TARGETS_TO_BUILD=host \
    externals/llvm-project/llvm

cmake --build build

