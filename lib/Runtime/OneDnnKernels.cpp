#include "dnnl.h"
#include "dnnl.hpp"
#include <cassert>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
using namespace dnnl;
using tag = memory::format_tag;
using dt = memory::data_type;

template <typename T, size_t Num_Dims> struct tensor {
  T *buffer;                  // Allocated memory, used for deallocation only
  T *data;                    // Aligned data.
  intptr_t offset;            // Offset (in elems) to the first element
  intptr_t sizes[Num_Dims];   // Shape in elems
  intptr_t strides[Num_Dims]; // Strides in elems

  T &ref(int64_t n, int64_t c, int64_t h, int64_t w) {
    return *(data + n * strides[0] + c * strides[1] + h * strides[2] +
             2 * strides[3]);
  }
};

extern "C" void linalg_matmul_blas_f32(int64_t lhs_rank, tensor<float, 2> *lhs,
                                       int64_t rhs_rank, tensor<float, 2> *rhs,
                                       int64_t res_rank,
                                       tensor<float, 2> *res) {
  float alpha = 1.0;
  float beta = 1.0;
  auto m = lhs->sizes[0];
  auto k = lhs->sizes[1];
  auto n = rhs->sizes[1];

  // Prepare leading dimensions
  int64_t lda = k; // tolower(transA) == 'n' ? K : M;
  int64_t ldb = n; // tolower(transB) == 'n' ? N : K;
  int64_t ldc = n;

  dnnl_sgemm('n', 'n', m, n, k, alpha, lhs->data, lda, rhs->data, ldb, beta,
             res->data, ldc);
}

template <typename T, size_t Num_Dims>
void print_tensor(tensor<T, Num_Dims> *t, std::vector<int> dims) {
  std::stringstream ss;
  ss.setf(std::ios::fixed);
  ss.precision(3);
  ss.width(6);
  ss << "matrix from {n: " << dims[0] << ", c: " << dims[1]
     << ", h: " << dims[2] << ", w: " << dims[3] << "}\n";
  ss << "[ \n";
  for (int64_t h = 0; h < dims[2]; h++) {
    ss << "  [ ";
    for (int64_t w = 0; w < dims[3]; w++) {
      ss << t->ref(dims[0], dims[1], h, w) << " ";
    }
    ss << "] \n";
  }
  ss << "]";
  std::cout << ss.str() << std::endl;
}

extern "C" void conv_f32(int32_t stride_h, int32_t stride_w, int64_t lhs_rank,
                         tensor<float, 4> *in, int64_t rhs_rank,
                         tensor<float, 4> *w, int64_t res_rank,
                         tensor<float, 4> *out) {

  // dnnl_set_verbose(2);
  // Create execution dnnl::engine.
  dnnl::engine engine(dnnl::engine::kind::cpu, 0);
  // Create dnnl::stream.
  dnnl::stream engine_stream(engine);
  // Tensor dimensions.
  const memory::dim N = in->sizes[0], // batch size
      IC = in->sizes[1],              // input channels
      IH = in->sizes[2],              // input height
      IW = in->sizes[3],              // input width
      OC = out->sizes[1],             // output channels
      KH = w->sizes[2],               // weights height
      KW = w->sizes[3],               // weights width
      PH_L = 0,                       // height padding: left
      PH_R = 0,                       // height padding: right
      PW_L = 0,                       // width padding: left
      PW_R = 0,                       // width padding: right
      SH = stride_h,                  // height-wise stride
      SW = stride_w,                  // width-wise stride
      OH = out->sizes[2],             // output height
      OW = out->sizes[3];             // output width
  // assert(OH == out->sizes[2]); // (IH - KH + PH_L + PH_R) / SH + 1
  // assert(OW == out->sizes[3]); // (IW - KW + PW_L + PW_R) / SW + 1

  // Source (src), weights, bias, and destination (dst) tensors
  // dimensions.
  memory::dims src_dims = {N, IC, IH, IW};
  memory::dims weights_dims = {OC, IC, KH, KW};
  memory::dims bias_dims = {OC};
  memory::dims dst_dims = {N, OC, OH, OW};
  // Strides, padding dimensions.
  memory::dims strides_dims = {SH, SW};
  memory::dims padding_dims_l = {PH_L, PW_L};
  memory::dims padding_dims_r = {PH_R, PW_R};

  // Create memory objects for tensor data (src, weights, dst). In this
  // example, NCHW layout is assumed for src and dst, and OIHW for weights.
  auto user_src_mem =
      memory({src_dims, dt::f32, tag::nchw}, engine); //, in->data);
  auto user_weights_mem =
      memory({weights_dims, dt::f32, tag::oihw}, engine); //, w->data);
  auto user_dst_mem =
      memory({dst_dims, dt::f32, tag::nchw}, engine); //, out->data);

  // print_tensor(in, {0, 0, 10, 20});
  // print_tensor(in, {0, 1, 10, 20});

  // print_tensor(w, {0, 0, 3, 3});
  // print_tensor(w, {9, 1, 3, 3});

  // Write data to memory object's handle.
  user_src_mem.set_data_handle(in->data);
  user_weights_mem.set_data_handle(w->data);
  user_dst_mem.set_data_handle(out->data);

  // Create memory descriptors with format_tag::any for the primitive. This
  // enables the convolution primitive to choose memory layouts for an
  // optimized primitive implementation, and these layouts may differ from the
  // ones provided by the user.
  auto conv_src_md = memory::desc(src_dims, dt::f32, tag::nchw);
  auto conv_weights_md = memory::desc(weights_dims, dt::f32, tag::oihw);
  auto conv_dst_md = memory::desc(dst_dims, dt::f32, tag::nchw);
  // Create memory descriptor and memory object for input bias.
  auto user_bias_md = memory::desc(bias_dims, dt::f32, tag::a);
  auto user_bias_mem = memory(user_bias_md, engine);
  // float zero_b[bias_dims[0]] = {};
  // user_bias_mem.set_data_handle(zero_b);

  auto zero_b = std::make_unique<float[]>(bias_dims[0]);
  user_bias_mem.set_data_handle(zero_b.get());

  // Create primitive post-ops (ReLU).
  // const float alpha = 0.f;
  // const float beta = 0.f;
  post_ops conv_ops;
  // conv_ops.append_eltwise(algorithm::eltwise_relu, alpha, beta);
  primitive_attr conv_attr;
  // conv_attr.set_post_ops(conv_ops);

  // Create primitive descriptor.
  auto conv_pd = convolution_forward::primitive_desc(
      engine, prop_kind::forward_inference, algorithm::convolution_direct,
      conv_src_md, conv_weights_md, user_bias_md, conv_dst_md, strides_dims,
      padding_dims_l, padding_dims_r, conv_attr);

  // For now, assume that the src, weights, and dst memory layouts generated
  // by the primitive and the ones provided by the user are identical.
  auto conv_src_mem = user_src_mem;
  auto conv_weights_mem = user_weights_mem;
  auto conv_dst_mem = user_dst_mem;

  // Reorder the data in case the src and weights memory layouts generated by
  // the primitive and the ones provided by the user are different. In this
  // case, we create additional memory objects with internal buffers that will
  // contain the reordered data. The data in dst will be reordered after the
  // convolution computation has finalized.
  // if (conv_pd.src_desc() != user_src_mem.get_desc()) {
  //   conv_src_mem = memory(conv_pd.src_desc(), engine);
  //   reorder(user_src_mem, conv_src_mem)
  //       .execute(engine_stream, user_src_mem, conv_src_mem);
  // }
  // if (conv_pd.weights_desc() != user_weights_mem.get_desc()) {
  //   conv_weights_mem = memory(conv_pd.weights_desc(), engine);
  //   reorder(user_weights_mem, conv_weights_mem)
  //       .execute(engine_stream, user_weights_mem, conv_weights_mem);
  // }

  // if (conv_pd.dst_desc() != user_dst_mem.get_desc()) {
  //   conv_dst_mem = memory(conv_pd.dst_desc(), engine);
  // }

  // Create the primitive.
  auto conv_prim = convolution_forward(conv_pd);
  // Primitive arguments.
  std::unordered_map<int, memory> conv_args;
  conv_args.insert({DNNL_ARG_SRC, conv_src_mem});
  conv_args.insert({DNNL_ARG_WEIGHTS, conv_weights_mem});
  conv_args.insert({DNNL_ARG_BIAS, user_bias_mem});
  conv_args.insert({DNNL_ARG_DST, conv_dst_mem});

  // Primitive execution: convolution with ReLU.
  conv_prim.execute(engine_stream, conv_args);

  // Reorder the data in case the dst memory descriptor generated by the
  // primitive and the one provided by the user are different.
  // if (conv_pd.dst_desc() != user_dst_mem.get_desc()) {
  //   reorder(conv_dst_mem, user_dst_mem)
  //       .execute(engine_stream, conv_dst_mem, user_dst_mem);
  // } else {
  //   user_dst_mem = conv_dst_mem;
  // }
  user_dst_mem = conv_dst_mem;

  // Wait for the computation to finalize.
  engine_stream.wait();

  // print_tensor(out, {0, 0, 8, 18});
  // print_tensor(out, {0, 9, 8, 18});
}
