#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>

#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#define THRESH_HOLD 15.0f

__device__ __forceinline__ float softplus_sqrt_f32(float x) {
    float sp;
    if (x > THRESH_HOLD) {
        sp = x;
    } else if (x < -THRESH_HOLD) {
        sp = __expf(x);
    } else {
        sp = log1pf(__expf(x));
    }
    return sqrtf(sp);
}

__device__ __forceinline__ float softplus_sqrt_f16_scalar(float x) {
    return softplus_sqrt_f32(x);
}


__global__ void softplus_sqrt_f16_kernel(
    const __half* __restrict__ in,
    __half*       __restrict__ out,
    int64_t numel)
{
    constexpr int VEC = 8;
    int64_t vec_numel = numel / VEC;
    int64_t idx    = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = gridDim.x * blockDim.x;

    const float4* in4  = reinterpret_cast<const float4*>(in);
    float4*       out4 = reinterpret_cast<float4*>(out);

    for (int64_t i = idx; i < vec_numel; i += stride) {
        float4 v = in4[i];

        __half2 h0 = reinterpret_cast<const __half2*>(&v)[0];
        __half2 h1 = reinterpret_cast<const __half2*>(&v)[1];
        __half2 h2 = reinterpret_cast<const __half2*>(&v)[2];
        __half2 h3 = reinterpret_cast<const __half2*>(&v)[3];

        float4 res;
        reinterpret_cast<__half2*>(&res)[0] = __floats2half2_rn(
            softplus_sqrt_f16_scalar(__half2float(h0.x)),
            softplus_sqrt_f16_scalar(__half2float(h0.y)));
        reinterpret_cast<__half2*>(&res)[1] = __floats2half2_rn(
            softplus_sqrt_f16_scalar(__half2float(h1.x)),
            softplus_sqrt_f16_scalar(__half2float(h1.y)));
        reinterpret_cast<__half2*>(&res)[2] = __floats2half2_rn(
            softplus_sqrt_f16_scalar(__half2float(h2.x)),
            softplus_sqrt_f16_scalar(__half2float(h2.y)));
        reinterpret_cast<__half2*>(&res)[3] = __floats2half2_rn(
            softplus_sqrt_f16_scalar(__half2float(h3.x)),
            softplus_sqrt_f16_scalar(__half2float(h3.y)));

        out4[i] = res;
    }

    int64_t tail_start = vec_numel * VEC;
    for (int64_t i = tail_start + idx; i < numel; i += stride) {
        out[i] = __float2half(
            softplus_sqrt_f16_scalar(__half2float(in[i])));
    }
}

torch::Tensor softplus_sqrt_cuda(torch::Tensor input) {
    int64_t numel = input.numel();

    if (numel <= 1024 * 1024) {
        return at::sqrt(at::softplus(input));
    }
    
    auto output = torch::empty_like(input);
    const int threads = 256;
    int64_t vec_numel = numel / 8;
    int64_t blocks = (vec_numel + threads - 1) / threads;
    blocks = std::max(blocks, (int64_t)1);
    blocks = std::min(blocks, (int64_t)65535);
    softplus_sqrt_f16_kernel<<<blocks, threads>>>(
        reinterpret_cast<const __half*>(input.data_ptr<at::Half>()),
        reinterpret_cast<__half*>(output.data_ptr<at::Half>()),
        numel);

    return output;
}
