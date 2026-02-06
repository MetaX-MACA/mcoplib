// 2025 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
#include <torch/extension.h>
#include <torch/torch.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CachingHostAllocator.h>
#include <c10/cuda/CUDAGuard.h>
/* Includes, cuda */
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include "mcoplib_ops_params_info.hpp"
#include "mcoplib_ops_params_dump.hpp"
namespace grouped_gemm {
#define NUM_STREAM 4
// cublasStatus_t cublasGemmGroupedBatchedEx(cublasHandle_t handle,
//                             const cublasOperation_t transa_array[],
//                             const cublasOperation_t transb_array[],
//                             const int m_array[],
//                             const int n_array[],
//                             const int k_array[],
//                             const void    *alpha_array,
//                             const void     *const Aarray[],
//                             cudaDataType_t Atype,
//                             const int lda_array[],
//                             const void     *const Barray[],
//                             cudaDataType_t Btype,
//                             const int ldb_array[],
//                             const void    *beta_array,
//                             void           *const Carray[],
//                             cudaDataType_t Ctype,
//                             const int ldc_array[],
//                             int group_count,
//                             const int group_size[],
//                             cublasComputeType_t computeType);
// single stream cublasGemmGroupedBatchedEx
// trans_a == False
template<typename arg_t, cudaDataType_t cuda_t>
void CublasGroupedGemm(torch::Tensor a,
           torch::Tensor b,
           torch::Tensor c,
           torch::Tensor batch_sizes,
           bool trans_a,
           bool trans_b) {
  cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
  c10::cuda::CUDAStream stream = c10::cuda::getCurrentCUDAStream();
  const int group_count = batch_sizes.size(0);
  int64_t n = trans_b ? b.size(1) : b.size(2);
  int64_t k = a.size(1);
  int64_t b_rows = b.size(1), b_cols = b.size(2);
  arg_t* a_ptr = a.data_ptr<arg_t>();
  arg_t* b_ptr = b.data_ptr<arg_t>();
  arg_t* c_ptr = c.data_ptr<arg_t>();
  int m_array[group_count], n_array[group_count], k_array[group_count];
  float alpha_array[group_count], beta_array[group_count];
  int group_size[group_count];
  cublasOperation_t transa_array[group_count];
  cublasOperation_t transb_array[group_count];
  int lda_array[group_count];
  int ldb_array[group_count];
  int ldc_array[group_count];
  void **A_array_host; 
  void **B_array_host;
  void **C_array_host;
  auto host_space = at::cuda::HostAlloc(3 * group_count * sizeof(void *));
  A_array_host = (void**)(host_space.get());
  B_array_host = (void**)((uint8_t*)host_space.get() + group_count * sizeof(void *));
  C_array_host = (void**)((uint8_t*)host_space.get() + 2 * group_count * sizeof(void *));
  void **A_array_dev;
  void **B_array_dev;
  void **C_array_dev;
  auto& dev_allocator = *c10::cuda::CUDACachingAllocator::get();
  auto dev_space = dev_allocator.allocate(3 * group_count * sizeof(void *));
  A_array_dev = (void**)(dev_space.get());
  B_array_dev = (void**)((uint8_t*)dev_space.get() + group_count * sizeof(void *));
  C_array_dev = (void**)((uint8_t*)dev_space.get() + 2 * group_count * sizeof(void *));
  for (int i=0; i<group_count; i++) {
    transa_array[i] = trans_a ? CUBLAS_OP_T : CUBLAS_OP_N;
    transb_array[i] = trans_b ? CUBLAS_OP_T : CUBLAS_OP_N;
    int64_t m = batch_sizes.data_ptr<int64_t>()[i];
    m_array[i] = m;
    n_array[i] = n; 
    k_array[i] = k;
    A_array_host[i] = reinterpret_cast<void*>(a_ptr);
    B_array_host[i] = reinterpret_cast<void*>(b_ptr);
    C_array_host[i] = reinterpret_cast<void*>(c_ptr);
    lda_array[i] = trans_a ? m : k;
    ldb_array[i] = trans_b ? k : n;
    ldc_array[i] = n;
    alpha_array[i] = 1.0;
    beta_array[i] = 0.0;
    group_size[i] = 1;
    a_ptr += m * k;
    b_ptr += b_rows * b_cols;
    c_ptr += m * n;
  }
  AT_CUDA_CHECK(cudaMemcpyAsync(A_array_dev, A_array_host, group_count * sizeof(void *), cudaMemcpyHostToDevice, (cudaStream_t)stream));
  AT_CUDA_CHECK(cudaMemcpyAsync(B_array_dev, B_array_host, group_count * sizeof(void *), cudaMemcpyHostToDevice, (cudaStream_t)stream));
  AT_CUDA_CHECK(cudaMemcpyAsync(C_array_dev, C_array_host, group_count * sizeof(void *), cudaMemcpyHostToDevice, (cudaStream_t)stream));
  auto* ctx = host_space.get_context();
  at::cuda::CachingHostAllocator_recordEvent(A_array_host, ctx, stream);
  at::cuda::CachingHostAllocator_recordEvent(B_array_host, ctx, stream);
  at::cuda::CachingHostAllocator_recordEvent(C_array_host, ctx, stream);
  mcblasGemmGroupedBatchedEx(handle,
                            transb_array, // transb
                            transa_array, // False
                            n_array,
                            m_array,
                            k_array,
                            alpha_array,  // 1.0
                            B_array_dev,
                            cuda_t,
                            ldb_array,    // k or n
                            A_array_dev, 
                            cuda_t,
                            lda_array,    //k
                            beta_array,   // 0.0
                            C_array_dev,
                            cuda_t,
                            ldc_array,    // n
                            group_count,
                            group_size,   // 1
                            CUBLAS_COMPUTE_32F);
}
// cublasStatus_t cublasGemmEx(cublasHandle_t handle,
//                            cublasOperation_t transa,
//                            cublasOperation_t transb,
//                            int m,
//                            int n,
//                            int k,
//                            const void    *alpha,
//                            const void     *A,
//                            cudaDataType_t Atype,
//                            int lda,
//                            const void     *B,
//                            cudaDataType_t Btype,
//                            int ldb,
//                            const void    *beta,
//                            void           *C,
//                            cudaDataType_t Ctype,
//                            int ldc,
//                            cublasComputeType_t computeType,
//                            cublasGemmAlgo_t algo)
// multi stream mcblasSgemm for grouped gemm
// trans_a == False
template<typename arg_t, cudaDataType_t cuda_t>
void CublasGroupedGemmMultiStream(torch::Tensor a,
           torch::Tensor b,
           torch::Tensor c,
           torch::Tensor batch_sizes,
           bool trans_a,
           bool trans_b) {
    // init stream
    mcStream_t multiStreams[NUM_STREAM];
    mcEvent_t multiStreamEvents[NUM_STREAM];
    mcEvent_t curStreamEvent;
    cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
    c10::cuda::CUDAStream curStream = c10::cuda::getCurrentCUDAStream();
    mcEventCreateWithFlags(&curStreamEvent, mcEventDisableTiming);
    for (int i=0; i < NUM_STREAM; i++) {
      multiStreams[i] = c10::cuda::getStreamFromPool();
      mcEventCreateWithFlags(&multiStreamEvents[i], mcEventDisableTiming);
    }
    // sync stream
    // multiStreamWaitCurrentStream
    mcEventRecord(curStreamEvent, curStream);
    for (int i = 0; i < NUM_STREAM; ++i) {
        mcStreamWaitEvent(multiStreams[i], curStreamEvent);
    }
    const int group_count = batch_sizes.size(0);
    arg_t* a_ptr = a.data_ptr<arg_t>();
    arg_t* b_ptr = b.data_ptr<arg_t>();
    arg_t* c_ptr = c.data_ptr<arg_t>();
    int n = trans_b ? b.size(1) : b.size(2);
    int k = a.size(1);
    int b_rows = b.size(1), b_cols = b.size(2);
    int ldb = trans_b ? k : n;
    int ldc = n;
    cublasOperation_t transa = trans_a ? CUBLAS_OP_T : CUBLAS_OP_N;
    cublasOperation_t transb = trans_b ? CUBLAS_OP_T : CUBLAS_OP_N;
    float alpha = 1.0;
    float beta = 0;
    for (int i = 0; i < group_count; ++i) {
      int m = batch_sizes.data_ptr<int64_t>()[i];
      int lda = trans_a ? m : k;
      mcblasSetStream(handle, multiStreams[i % NUM_STREAM]);
      mcblasGemmEx(handle,
                  transb, transa,
                  n, m, k,
                  reinterpret_cast<void*>(&alpha),
                  reinterpret_cast<void*>(b_ptr), cuda_t, ldb,
                  reinterpret_cast<void*>(a_ptr), cuda_t, lda,
                  reinterpret_cast<void*>(&beta),
                  reinterpret_cast<void*>(c_ptr), cuda_t, ldc,
                  CUBLAS_COMPUTE_32F,
                  MCBLAS_GEMM_DEFAULT);
      a_ptr += m * k;
      b_ptr += b_rows * b_cols;
      c_ptr += m * n;
    }
    // sync stream
    // currentStreamWaitMultiStream
    for (int i = 0; i < NUM_STREAM; ++i) {
        mcEventRecord(multiStreamEvents[i], multiStreams[i]);
    }
    for (int i = 0; i < NUM_STREAM; ++i) {
        mcStreamWaitEvent(curStream, multiStreamEvents[i]);
    }
    mcEventDestroy(curStreamEvent);
    for (int i=0; i < NUM_STREAM; i++) {
      mcEventDestroy(multiStreamEvents[i]);
    }
}
// trans_a == True, transb == False
template<typename arg_t, cudaDataType_t cuda_t>
void CublasGroupedGemmVariableK(torch::Tensor a,
        torch::Tensor b,
        torch::Tensor c,
        torch::Tensor batch_sizes) {
  cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
  c10::cuda::CUDAStream stream = c10::cuda::getCurrentCUDAStream();
  const int group_count = batch_sizes.size(0);
  int64_t n = b.size(1);
  int64_t m = a.size(1);
  arg_t* a_ptr = a.data_ptr<arg_t>();
  arg_t* b_ptr = b.data_ptr<arg_t>();
  arg_t* c_ptr = c.data_ptr<arg_t>();
  int m_array[group_count], n_array[group_count], k_array[group_count];
  float alpha_array[group_count], beta_array[group_count];
  int group_size[group_count];
  cublasOperation_t transa_array[group_count];
  cublasOperation_t transb_array[group_count];
  int lda_array[group_count];
  int ldb_array[group_count];
  int ldc_array[group_count];
  void **A_array_host; 
  void **B_array_host;
  void **C_array_host;
  auto host_space = at::cuda::HostAlloc(3 * group_count * sizeof(void *));
  A_array_host = (void**)(host_space.get());
  B_array_host = (void**)((uint8_t*)host_space.get() + group_count * sizeof(void *));
  C_array_host = (void**)((uint8_t*)host_space.get() + 2 * group_count * sizeof(void *));
  void **A_array_dev;
  void **B_array_dev;
  void **C_array_dev;
  auto& allocator = *c10::cuda::CUDACachingAllocator::get();
  auto dev_space = allocator.allocate(3 * group_count * sizeof(void *));
  A_array_dev = (void**)(dev_space.get());
  B_array_dev = (void**)((uint8_t*)dev_space.get() + group_count * sizeof(void *));
  C_array_dev = (void**)((uint8_t*)dev_space.get() + 2 * group_count * sizeof(void *));
  for (int i=0; i<group_count; i++) {
    transa_array[i] = CUBLAS_OP_T;
    transb_array[i] = CUBLAS_OP_N;
    int64_t k = batch_sizes.data_ptr<int64_t>()[i];
    m_array[i] = m;
    n_array[i] = n; 
    k_array[i] = k;
    A_array_host[i] = reinterpret_cast<void*>(a_ptr);
    B_array_host[i] = reinterpret_cast<void*>(b_ptr);
    C_array_host[i] = reinterpret_cast<void*>(c_ptr);
    lda_array[i] = m;
    ldb_array[i] = n;
    ldc_array[i] = n;
    alpha_array[i] = 1.0;
    beta_array[i] = 0.0;
    group_size[i] = 1;
    a_ptr += k * m;
    b_ptr += k * n;
    c_ptr += m * n;
  }
  AT_CUDA_CHECK(cudaMemcpyAsync(A_array_dev, A_array_host, group_count * sizeof(void *), cudaMemcpyHostToDevice, (cudaStream_t)stream));
  AT_CUDA_CHECK(cudaMemcpyAsync(B_array_dev, B_array_host, group_count * sizeof(void *), cudaMemcpyHostToDevice, (cudaStream_t)stream));
  AT_CUDA_CHECK(cudaMemcpyAsync(C_array_dev, C_array_host, group_count * sizeof(void *), cudaMemcpyHostToDevice, (cudaStream_t)stream));
  auto* ctx = host_space.get_context();
  at::cuda::CachingHostAllocator_recordEvent(A_array_host, ctx, stream);
  at::cuda::CachingHostAllocator_recordEvent(B_array_host, ctx, stream);
  at::cuda::CachingHostAllocator_recordEvent(C_array_host, ctx, stream);
  mcblasGemmGroupedBatchedEx(handle,
                            transb_array,  // False
                            transa_array,  // True
                            n_array,
                            m_array,
                            k_array,
                            alpha_array,  // 1
                            B_array_dev,
                            cuda_t,
                            ldb_array,    // n
                            A_array_dev,
                            cuda_t,
                            lda_array,    // m
                            beta_array,   // 0
                            C_array_dev,
                            cuda_t,
                            ldc_array,    // n
                            group_count,
                            group_size,   // 1
                            CUBLAS_COMPUTE_32F);
}
template<typename arg_t, cudaDataType_t cuda_t>
void CublasGroupedGemmVariableKMultiStream(torch::Tensor a,
        torch::Tensor b,
        torch::Tensor c,
        torch::Tensor batch_sizes) {
    // init stream 
    mcStream_t multiStreams[NUM_STREAM];
    mcEvent_t multiStreamEvents[NUM_STREAM];
    mcEvent_t curStreamEvent;
    cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
    c10::cuda::CUDAStream curStream = c10::cuda::getCurrentCUDAStream();
    mcEventCreateWithFlags(&curStreamEvent, mcEventDisableTiming);
    for (int i=0; i < NUM_STREAM; i++) {
      multiStreams[i] = c10::cuda::getStreamFromPool();

      mcEventCreateWithFlags(&multiStreamEvents[i], mcEventDisableTiming);
    }
    // sync stream
    // multiStreamWaitCurrentStream
    mcEventRecord(curStreamEvent, curStream);
    for (int i = 0; i < NUM_STREAM; ++i) {
        mcStreamWaitEvent(multiStreams[i], curStreamEvent);
    }
    const int group_count = batch_sizes.size(0);
    arg_t* a_ptr = a.data_ptr<arg_t>();
    arg_t* b_ptr = b.data_ptr<arg_t>();
    arg_t* c_ptr = c.data_ptr<arg_t>();
    int n = b.size(1);
    int m = a.size(1);
    int lda = m;
    int ldb = n;
    int ldc = n;
    cublasOperation_t transa = CUBLAS_OP_T;
    cublasOperation_t transb = CUBLAS_OP_N;
    float alpha = 1.0;
    float beta = 0;
    for (int i = 0; i < group_count; ++i) {
      int k = batch_sizes.data_ptr<int64_t>()[i];
      mcblasSetStream(handle, multiStreams[i % NUM_STREAM]);
      mcblasGemmEx(handle,
                  transb, transa,
                  n, m, k,
                  reinterpret_cast<void*>(&alpha),
                  reinterpret_cast<void*>(b_ptr), cuda_t, ldb,
                  reinterpret_cast<void*>(a_ptr), cuda_t, lda,
                  reinterpret_cast<void*>(&beta),
                  reinterpret_cast<void*>(c_ptr), cuda_t, ldc,
                  CUBLAS_COMPUTE_32F,
                  MCBLAS_GEMM_DEFAULT);
      
      a_ptr += k * m;
      b_ptr += k * n;
      c_ptr += m * n;
    }
    // sync stream
    for (int i = 0; i < NUM_STREAM; ++i) {
        mcEventRecord(multiStreamEvents[i], multiStreams[i]);
    }
    for (int i = 0; i < NUM_STREAM; ++i) {
        mcStreamWaitEvent(curStream, multiStreamEvents[i]);
    }
    mcEventDestroy(curStreamEvent);
    for (int i=0; i < NUM_STREAM; i++) {
      mcEventDestroy(multiStreamEvents[i]);
    }
}
void GroupedGemmVariableK(torch::Tensor a,
        torch::Tensor b,
        torch::Tensor c,
        torch::Tensor batch_sizes) {
  // We expected a CUDA tensor with two dimensions and shape
  // (tokens, hidden_out) for 'b'.
  TORCH_CHECK(b.is_cuda());
  TORCH_CHECK(b.ndimension() == 2);
  TORCH_CHECK(b.scalar_type() == torch::kBFloat16 || b.scalar_type() == torch::kHalf || b.scalar_type() == torch::kFloat);
  // Validate the dimensions.
  int64_t tokens = a.size(0), num_experts = batch_sizes.size(0);
  int64_t m = a.size(1), n = b.size(1);
  // Validate that we have the same contraction dimension.
  TORCH_CHECK(tokens == b.size(0));
  // Validate the output shape.
  TORCH_CHECK(c.is_cuda());
  TORCH_CHECK(c.ndimension() == 3);
  TORCH_CHECK(c.scalar_type() == torch::kBFloat16 || c.scalar_type() == torch::kHalf || c.scalar_type() == torch::kFloat);
  TORCH_CHECK(c.size(0) == num_experts);
  TORCH_CHECK(c.size(1) == m);
  TORCH_CHECK(c.size(2) == n);
  TORCH_CHECK(a.scalar_type() == b.scalar_type() && a.scalar_type() == c.scalar_type());
  // Run the computation.
  if (a.scalar_type() == torch::kBFloat16) {
    // CublasGroupedGemmVariableK<c10::BFloat16, CUDA_R_16BF>(a, b, c, batch_sizes);
    CublasGroupedGemmVariableKMultiStream<c10::BFloat16, CUDA_R_16BF>(a, b, c, batch_sizes);
  } else if (a.scalar_type() == torch::kHalf) {
    // CublasGroupedGemmVariableK<c10::Half, CUDA_R_16F>(a, b, c, batch_sizes);
    CublasGroupedGemmVariableKMultiStream<c10::Half, CUDA_R_16F>(a, b, c, batch_sizes);
  } else if (a.scalar_type() == torch::kFloat) {
    CublasGroupedGemmVariableKMultiStream<float, CUDA_R_32F>(a, b, c, batch_sizes);
  } else {
    assert(0);
  }
}
void GroupedGemm(torch::Tensor a,
     torch::Tensor b,
     torch::Tensor c,
     torch::Tensor batch_sizes,
     bool trans_a, bool trans_b) {
  DEBUG_TRACE_PARAMS(a, b, c, batch_sizes, trans_a, trans_b);
  DEBUG_DUMP_PARAMS(a, b, c, batch_sizes, trans_a, trans_b);
  // NOTE: We only support 'trans_a' or 'trans_b', not both.
  TORCH_CHECK(!(trans_a && trans_b));
  // We expect the batch_sizes on CPU.
  TORCH_CHECK(batch_sizes.is_cpu());
  TORCH_CHECK(batch_sizes.ndimension() == 1);
  TORCH_CHECK(batch_sizes.scalar_type() == torch::kInt64);
  // We expected a CUDA tensor with two dimensions and shape
  // (tokens, hidden_in) for 'a'.
  TORCH_CHECK(a.is_cuda());
  TORCH_CHECK(a.ndimension() == 2);
  TORCH_CHECK(a.scalar_type() == torch::kBFloat16 || a.scalar_type() == torch::kHalf || a.scalar_type() == torch::kFloat);
  // TORCH_CHECK(!trans_a)
  if (trans_a) {
    GroupedGemmVariableK(a, b, c, batch_sizes);
    return;
  }
  // We expected a CUDA tensor with three dimensions and shape
  // (num_experts, hidden_in, hidden_out) for 'b'.
  TORCH_CHECK(b.is_cuda());
  TORCH_CHECK(b.ndimension() == 3);
  TORCH_CHECK(b.scalar_type() == torch::kBFloat16 || b.scalar_type() == torch::kHalf || b.scalar_type() == torch::kFloat);
  // Validate the contraction dimensions match.
  int64_t tokens = a.size(0), num_experts = b.size(0);
  int64_t hidden_in = trans_b ? b.size(2) : b.size(1);
  int64_t hidden_out = trans_b ? b.size(1) : b.size(2);
  TORCH_CHECK(hidden_in == a.size(1));
  // Validate that we have one size per expert.
  TORCH_CHECK(batch_sizes.size(0) == num_experts);
  // Validate the output shape.
  TORCH_CHECK(c.is_cuda());
  TORCH_CHECK(c.ndimension() == 2);
  TORCH_CHECK(c.scalar_type() == torch::kBFloat16 || c.scalar_type() == torch::kHalf || c.scalar_type() == torch::kFloat);
  TORCH_CHECK(c.size(0) == tokens);
  TORCH_CHECK(c.size(1) == hidden_out);
  // NOTE: We support transposition through the 'trans_b' flag.
  TORCH_CHECK(a.is_contiguous());
  TORCH_CHECK(b.is_contiguous());
  TORCH_CHECK(a.scalar_type() == b.scalar_type() && a.scalar_type() == c.scalar_type());
  // NOTE: Use cuBLAS for SM90 until CUTLASS supports SM90-optimized grouped-gemm.
  if (a.scalar_type() == torch::kBFloat16) {
    // CublasGroupedGemm<c10::BFloat16, CUDA_R_16BF>(a, b, c, batch_sizes, trans_a, trans_b);
    CublasGroupedGemmMultiStream<c10::BFloat16, CUDA_R_16BF>(a, b, c, batch_sizes, trans_a, trans_b);
  } else if (a.scalar_type() == torch::kHalf) {
    // CublasGroupedGemm<c10::Half, CUDA_R_16F>(a, b, c, batch_sizes, trans_a, trans_b);
    CublasGroupedGemmMultiStream<c10::Half, CUDA_R_16F>(a, b, c, batch_sizes, trans_a, trans_b);
  } else if (a.scalar_type() == torch::kFloat) {
    CublasGroupedGemmMultiStream<float, CUDA_R_32F>(a, b, c, batch_sizes, trans_a, trans_b);
  } else {
    assert(0);
  }
  return;
}
}  // namespace grouped_gemm
