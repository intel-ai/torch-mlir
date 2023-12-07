#!/bin/bash

set -ex
cd $(dirname "$0")/../

project_dir=$PWD
echo "Using project dir: ${project_dir}"

# assuming git submodule update --init was done and all the code is present
pushd externals/tpp-mlir
git checkout -f main
git reset --hard
git apply ${project_dir}/utils/tpp-in-tree-build.patch
popd

pushd externals/llvm-project
rm -f llvm_version.txt
wget https://raw.githubusercontent.com/plaidml/tpp-mlir/main/build_tools/llvm_version.txt
git checkout -f `cat llvm_version.txt`
git reset --hard
popd


cmake -GNinja -Bbuild \
    -DCMAKE_BUILD_TYPE=Debug \
    -DPython3_FIND_VIRTUALENV=ONLY \
    -DLLVM_ENABLE_PROJECTS=mlir \
    -DLLVM_EXTERNAL_PROJECTS="torch-mlir;tpp-mlir" \
    -DLLVM_EXTERNAL_TORCH_MLIR_SOURCE_DIR="$PWD" \
    -DLLVM_EXTERNAL_TPP_MLIR_SOURCE_DIR="$PWD/externals/tpp-mlir" \
    -DMLIR_ENABLE_BINDINGS_PYTHON=ON \
    -DIMEX_ENABLE_L0_RUNTIME=ON \
    -DLLVM_TARGETS_TO_BUILD=host \
    externals/llvm-project/llvm

cmake --build build
