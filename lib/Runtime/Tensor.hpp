
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
