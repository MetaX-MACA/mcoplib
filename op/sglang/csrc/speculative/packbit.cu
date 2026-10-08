#include "pytorch_extension_utils.h"
#include "mcoplib_quantization.cuh"
#include "vec_dtypes.cuh"
using namespace mcoplib;

// bitorder = "little"
void segment_packbits(
    at::Tensor x,
    at::Tensor input_indptr,
    at::Tensor output_indptr,
    at::Tensor y,
    int64_t batch_size,
    int64_t cuda_stream) {
  CHECK_INPUT(x);
  CHECK_INPUT(input_indptr);
  CHECK_INPUT(output_indptr);
  auto device = x.device();
  CHECK_EQ(input_indptr.device(), device);
  CHECK_EQ(output_indptr.device(), device);
  CHECK_EQ(y.device(), device);
  CHECK_GE(output_indptr.size(0), batch_size + 1);

  cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  cudaError_t status = quantization::SegmentPackBits(
      static_cast<bool*>(x.data_ptr()),
      static_cast<uint8_t*>(y.data_ptr()),
      static_cast<int32_t*>(input_indptr.data_ptr()),
      static_cast<int32_t*>(output_indptr.data_ptr()),
      batch_size,
      quantization::BitOrder::kLittle,
      stream);
}
