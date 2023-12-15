#include "oneapi/dnnl/dnnl.hpp"

#include "Tensor.hpp"

#include <algorithm>
#include <cassert>
#include <cstring>
#include <iostream>
#include <numeric>

namespace {
inline dnnl::memory::dim product(const dnnl::memory::dims &dims) {
  return std::accumulate(dims.begin(), dims.end(), (dnnl::memory::dim)1,
                         std::multiplies<dnnl::memory::dim>());
}

inline void read_from_dnnl_memory(void *handle, dnnl::memory &mem) {
  dnnl::engine eng = mem.get_engine();
  size_t size = mem.get_desc().get_size();

  if (!handle)
    throw std::runtime_error("handle is nullptr.");

  if (eng.get_kind() == dnnl::engine::kind::gpu) {
    void *mapped_ptr = mem.map_data();
    if (mapped_ptr)
      std::memcpy(handle, mapped_ptr, size);
    mem.unmap_data(mapped_ptr);
    return;
  }

  if (eng.get_kind() == dnnl::engine::kind::cpu) {
    uint8_t *src = static_cast<uint8_t *>(mem.get_data_handle());
    if (!src)
      throw std::runtime_error("get_data_handle returned nullptr.");
    for (size_t i = 0; i < size; ++i)
      ((uint8_t *)handle)[i] = src[i];
    return;
  }

  assert(!"not expected");
}

inline void write_to_dnnl_memory(void *handle, dnnl::memory &mem) {
  dnnl::engine eng = mem.get_engine();
  size_t size = mem.get_desc().get_size();

  if (!handle)
    throw std::runtime_error("handle is nullptr.");

  if (eng.get_kind() == dnnl::engine::kind::gpu) {
    void *mapped_ptr = mem.map_data();
    if (mapped_ptr)
      std::memcpy(mapped_ptr, handle, size);
    mem.unmap_data(mapped_ptr);
    return;
  }

  if (eng.get_kind() == dnnl::engine::kind::cpu) {
    uint8_t *dst = static_cast<uint8_t *>(mem.get_data_handle());
    if (!dst)
      throw std::runtime_error("get_data_handle returned nullptr.");
    for (size_t i = 0; i < size; ++i)
      dst[i] = ((uint8_t *)handle)[i];
    return;
  }

  assert(!"not expected");
}
} // namespace

extern "C" {
void matmul_kernel_f32(int64_t lhs_rank, tensor<float> *lhs, int64_t rhs_rank,
                       tensor<float> *rhs, int64_t res_rank,
                       tensor<float> *res) {
  using namespace dnnl;
  const memory::dim m = lhs->rank[0];
  const memory::dim k = lhs->rank[1];
  const memory::dim n = rhs->rank[1];
  const memory::dim batch_size = 1;

  std::cout << "GPUs found: " << engine::get_count(engine::kind::gpu)
            << std::endl;
  engine eng(engine::kind::gpu, 0);
  stream strm(eng);

  memory::dims dims_lhs = {batch_size, m, k};
  memory::dims dims_rhs = {batch_size, k, n};
  memory::dims dims_res = {batch_size, m, n};

  using tag = memory::format_tag;
  using dt = memory::data_type;

  auto lhs_md = memory::desc(dims_lhs, dt::f32, tag::abc);
  auto rhs_md = memory::desc(dims_rhs, dt::f32, tag::abc);
  auto res_md = memory::desc(dims_res, dt::f32, tag::abc);

  auto lhs_mem = memory(lhs_md, eng);
  auto rhs_mem = memory(rhs_md, eng);
  auto res_mem = memory(res_md, eng);

  write_to_dnnl_memory(lhs->data, lhs_mem);
  write_to_dnnl_memory(rhs->data, rhs_mem);
  write_to_dnnl_memory(res->data, res_mem);

  auto matmul_pd = matmul::primitive_desc(eng, lhs_md, rhs_md, res_md);
  auto matmul_prim = matmul(matmul_pd);

  std::unordered_map<int, memory> matmul_args;
  matmul_args.insert({DNNL_ARG_SRC, lhs_mem});
  matmul_args.insert({DNNL_ARG_WEIGHTS, rhs_mem});
  matmul_args.insert({DNNL_ARG_DST, res_mem});

  matmul_prim.execute(strm, matmul_args);
  strm.wait();

  read_from_dnnl_memory(res->data, res_mem);
}
}
