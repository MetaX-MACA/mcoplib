#include <cuda_fp16.h>
#include "process_interface.h"

#include "mcoplib_ops_params_info.hpp"
#include "mcoplib_ops_params_dump.hpp"
// Y = 0.299*R + 0.587*G + 0.114*B
// U = -0.169*R - 0.331*G + 0.500*B + 0.5
// V = 0.500*R - 0.419*G - 0.081*B + 0.5

#define RGB2Y(R, G, B, Y) \
(Y) = 0.213 * (R) + 0.715 * (G) + 0.072 * (B);

#define RGB2UV(R , G, B, U, V) \
(U) = -0.117 * (R) - 0.394 * (G) + 0.511 * (B) + 128;\
(V) = 0.511 * (R) - 0.464 * (G) - 0.047 * (B) + 128;

template<int NUM_THREADS>
__global__ void fused_postprocess_nv12_Kernel(const float* __restrict inputData, uint8_t* __restrict nv12Data, int width, int height, float scale) 
{
    int u_h, u_w, y_h, y_w;
    int u_size = width * height;
    int y_stride = width << 1;
    int y_size = y_stride * height << 1;
    int index = blockIdx.x * NUM_THREADS + threadIdx.x;
    if(index >= u_size) return;
    u_h = index / width;
    u_w = index % width;
    y_h = u_h << 1;
    y_w = u_w << 1;
    uint8_t* ptr_y = nv12Data;
    uint8_t* ptr_uv = nv12Data + y_size;
    uint8_t* ptr_line0_y = ptr_y + y_h * y_stride + y_w;
    uint8_t* ptr_line1_y = ptr_line0_y + y_stride;
    uint8_t* ptr_line_uv = ptr_uv + u_h * y_stride + y_w;
    const float* ptr_r = inputData + y_h * y_stride + y_w;
    const float* ptr_g = ptr_r + y_size;
    const float* ptr_b = ptr_g + y_size;
    uint8_t y0[2]; 
    uint8_t y1[2];
    uint8_t uv[2];

    float r[2], g[2], b[2];
    int Y, U, V;
    *(float2*)r = *(float2*)ptr_r;
    *(float2*)g = *(float2*)ptr_g;
    *(float2*)b = *(float2*)ptr_b;
    int ri = roundf(r[0] * scale);
    int gi = roundf(g[0] * scale);
    int bi = roundf(b[0] * scale); 
    RGB2Y(ri, gi, bi, Y);
    Y = max(0, min(255, Y));

    RGB2UV(ri, gi, bi, U, V);
    U = max(0, min(255, U));
    V = max(0, min(255, V));

    y0[0] = static_cast<uint8_t>(Y);
    uv[0] = static_cast<uint8_t>(U);
    uv[1] = static_cast<uint8_t>(V);

    ri = roundf(r[1] * scale);
    gi = roundf(g[1] * scale);
    bi = roundf(b[1] * scale);
    RGB2Y(ri, gi, bi, Y);
    Y = max(0, min(255, Y));
    y0[1] = static_cast<uint8_t>(Y);
    *(uint16_t*)ptr_line0_y = *(uint16_t*)y0;
    *(uint16_t*)ptr_line_uv = *(uint16_t*)uv;

    *(float2*)r = *(float2*)(ptr_r + y_stride);
    *(float2*)g = *(float2*)(ptr_g + y_stride);
    *(float2*)b = *(float2*)(ptr_b + y_stride);

    ri = roundf(r[0] * scale);
    gi = roundf(g[0] * scale);
    bi = roundf(b[0] * scale);
    RGB2Y(ri, gi, bi, Y);
    Y = max(0, min(255, Y));
    y1[0] = static_cast<uint8_t>(Y);
    ri = roundf(r[1] * scale);
    gi = roundf(g[1] * scale);
    bi = roundf(b[1] * scale);
    RGB2Y(ri, gi, bi, Y);
    Y = max(0, min(255, Y));
    y1[1] = static_cast<uint8_t>(Y);
    *(uint16_t*)ptr_line1_y = *(uint16_t*)y1;
}

template<int NUM_THREADS>
__global__ void fused_postprocess_yuv420_Kernel(const float* __restrict inputData, uint8_t* __restrict yuv420Data, int width, int height, float scale) 
{
    int u_h, u_w, y_h, y_w;
    int u_size = width * height;
    int y_stride = width << 1;
    int y_size = y_stride * height << 1;
    int index = blockIdx.x * NUM_THREADS + threadIdx.x;
    u_h = index / width;
    u_w = index % width;
    y_h = u_h << 1;
    y_w = u_w << 1;
    uint8_t* ptr_y = yuv420Data;
    uint8_t* ptr_u = yuv420Data + y_size;
    uint8_t* ptr_v = ptr_u + u_size;
    uint8_t* ptr_line0_y = ptr_y + y_h * y_stride + y_w;
    uint8_t* ptr_line1_y = ptr_line0_y + y_stride;
    uint8_t* ptr_line_u = ptr_u + u_h * width + u_w;
    uint8_t* ptr_line_v = ptr_v + u_h * width + u_w;
    const float* ptr_r = inputData + y_h * y_stride + y_w;
    const float* ptr_g = ptr_r + y_size;
    const float* ptr_b = ptr_g + y_size;
    uint8_t y0[2]; 
    uint8_t y1[2];
    uint8_t uv[2];

    float r[2], g[2], b[2];
    int Y, U, V;
    *(float2*)r = *(float2*)ptr_r;
    *(float2*)g = *(float2*)ptr_g;
    *(float2*)b = *(float2*)ptr_b;
    int ri = roundf(r[0] * scale);
    int gi = roundf(g[0] * scale);
    int bi = roundf(b[0] * scale); 
    RGB2Y(ri, gi, bi, Y);
    Y = max(0, min(255, Y));
    RGB2UV(ri, gi, bi, U, V);
    U = max(0, min(255, U));
    V = max(0, min(255, V));

    y0[0] = static_cast<uint8_t>(Y);
    uv[0] = static_cast<uint8_t>(U);
    uv[1] = static_cast<uint8_t>(V);

    ri = roundf(r[1] * scale);
    gi = roundf(g[1] * scale);
    bi = roundf(b[1] * scale);
    RGB2Y(ri, gi, bi, Y);
    Y = max(0, min(255, Y));
    y0[1] = static_cast<uint8_t>(Y);
    *(uint16_t*)ptr_line0_y = *(uint16_t*)y0;
    *ptr_line_u = uv[0];
    *ptr_line_v = uv[1];
    *(float2*)r = *(float2*)(ptr_r + y_stride);
    *(float2*)g = *(float2*)(ptr_g + y_stride);
    *(float2*)b = *(float2*)(ptr_b + y_stride);

    ri = roundf(r[0] * scale);
    gi = roundf(g[0] * scale);
    bi = roundf(b[0] * scale);
    RGB2Y(ri, gi, bi, Y);
    Y = max(0, min(255, Y));
    y1[0] = static_cast<uint8_t>(Y);
    ri = roundf(r[1] * scale);
    gi = roundf(g[1] * scale);
    bi = roundf(b[1] * scale);
    RGB2Y(ri, gi, bi, Y);
    Y = max(0, min(255, Y));
    y1[1] = static_cast<uint8_t>(Y);
    *(uint16_t*)ptr_line1_y = *(uint16_t*)y1;
}

void fused_postprocess_nv12(const float* __restrict intputData, uint8_t* __restrict nv12Data, int width, int height, float scale, cudaStream_t stream)
{
    DEBUG_TRACE_PARAMS(intputData, nv12Data, width, height, scale, stream);
    DEBUG_DUMP_PARAMS(intputData, nv12Data, width, height, scale, stream);
    constexpr int num_threads = 512;
    int u_width = width / 2;
    int u_height = height / 2;
    int gridSize = (u_width * u_height + num_threads - 1) / num_threads;
    fused_postprocess_nv12_Kernel<num_threads><<<gridSize, num_threads, 0, stream>>>(intputData, nv12Data, u_width, u_height, scale);
}

void fused_postprocess_yuv420(const float* __restrict inputData, uint8_t* __restrict yuv420Data, int width, int height, float scale, cudaStream_t stream)
{
    DEBUG_TRACE_PARAMS(inputData, yuv420Data, width, height, scale, stream);
    DEBUG_DUMP_PARAMS(inputData, yuv420Data, width, height, scale, stream);
    constexpr int num_threads = 512;
    int u_width = width / 2;
    int u_height = height / 2;
    int gridSize = (u_width * u_height + num_threads - 1) / num_threads;
    fused_postprocess_yuv420_Kernel<num_threads><<<gridSize, num_threads, 0, stream>>>(inputData, yuv420Data, u_width, u_height, scale);
}