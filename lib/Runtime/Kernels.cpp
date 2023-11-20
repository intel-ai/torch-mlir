#include <cstdint>
#include <iostream>

#include "mkl.h"

template <typename T> struct tensor {
  T *buffer;         // Allocated memory, used for deallocation only
  T *data;           // Aligned data.
  int64_t offset;    // Offset (in elems) to the first element
  int64_t rank[2];   // Shape in elems
  int64_t stride[2]; // Strides in elems

  T &ref(int64_t i, int64_t j) {
    return *(data + i * stride[0] + j * stride[1]);
  }
};

namespace {
template <typename T>
void trace_matmul_call(int64_t lhs_rank, tensor<T> *lhs, int64_t rhs_rank,
                       tensor<T> *rhs, int64_t res_rank, tensor<T> *res) {
  std::cout << "test_matmul" << std::endl;
  std::cout << "  lhs_rank=" << lhs_rank << std::endl
            << "  rhs_rank=" << rhs_rank << std::endl
            << "  res_rank=" << res_rank << std::endl;
  std::cout << "  lhs={" << lhs->buffer << ", " << lhs->data << ", "
            << lhs->offset << ", [" << lhs->rank[0] << ", " << lhs->rank[1]
            << "], [" << lhs->stride[0] << ", " << lhs->stride[1] << "]}"
            << std::endl;
  std::cout << "  rhs={" << rhs->buffer << ", " << rhs->data << ", "
            << rhs->offset << ", [" << rhs->rank[0] << ", " << rhs->rank[1]
            << "], [" << rhs->stride[0] << ", " << rhs->stride[1] << "]}"
            << std::endl;
  std::cout << "  res={" << res->buffer << ", " << res->data << ", "
            << res->offset << ", [" << res->rank[0] << ", " << res->rank[1]
            << "], [" << res->stride[0] << ", " << res->stride[1] << "]}"
            << std::endl;
}

template <typename T>
void test_matmul(int64_t lhs_rank, tensor<T> *lhs, int64_t rhs_rank,
                 tensor<T> *rhs, int64_t res_rank, tensor<T> *res) {
  trace_matmul_call(lhs_rank, lhs, rhs_rank, rhs, res_rank, res);
  std::cout << "Using naive implementation." << std::endl;
  for (int64_t i = 0; i < lhs->rank[0]; ++i) {
    for (int64_t j = 0; j < rhs->rank[1]; ++j) {
      for (int64_t k = 0; k < rhs->rank[0]; ++k) {
        res->ref(i, j) += lhs->ref(i, k) * rhs->ref(k, j);
      }
    }
  }
}
} // namespace

// extern "C" void test_matmul_f16(int64_t lhs_rank, tensor<_Float16> *lhs,
//                                 int64_t rhs_rank, tensor<_Float16> *rhs,
//                                 int64_t res_rank, tensor<_Float16> *res) {
//   test_matmul(lhs_rank, lhs, rhs_rank, rhs, res_rank, res);
// }

extern "C" void matmul_kernel_f32(int64_t lhs_rank, tensor<float> *lhs,
                                  int64_t rhs_rank, tensor<float> *rhs,
                                  int64_t res_rank, tensor<float> *res) {
  // trace_matmul_call(lhs_rank, lhs, rhs_rank, rhs, res_rank, res);
  // std::cout << "Using MKL implementation." << std::endl;
  float alpha = 1.0;
  float beta = 1.0;
  auto m = lhs->rank[0];
  auto k = lhs->rank[1];
  auto n = rhs->rank[1];
  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, alpha,
              lhs->data, k, rhs->data, n, beta, res->data, n);
}

extern "C" void matmulb_kernel_f32(int64_t lhs_rank, tensor<float> *lhs,
                                   int64_t rhs_rank, tensor<float> *rhs,
                                   int64_t res_rank, tensor<float> *res) {
  // trace_matmul_call(lhs_rank, lhs, rhs_rank, rhs, res_rank, res);
  float alpha = 1.0;
  float beta = 1.0;
  auto m = lhs->rank[0];
  auto k = lhs->rank[1];
  auto n = rhs->rank[0];
  // std::cout << "Using MKL implementation. [m(rows op(A)): " << m
  //           << ", k(cols op(A)): " << k << ", n(cols op(B)  = cols C): " << n
  //           << "]" << std::endl;
  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, m, n, k, alpha,
              lhs->data, k, rhs->data, k, beta, res->data, n);
}

extern "C" void matmul_kernel_f64(int64_t lhs_rank, tensor<double> *lhs,
                                  int64_t rhs_rank, tensor<double> *rhs,
                                  int64_t res_rank, tensor<double> *res) {
  double alpha = 1.0;
  double beta = 1.0;
  auto m = lhs->rank[0];
  auto k = lhs->rank[1];
  auto n = rhs->rank[1];
  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, alpha,
              lhs->data, k, rhs->data, n, beta, res->data, n);
}
