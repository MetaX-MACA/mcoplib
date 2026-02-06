
#include <cuda_fp16.h>
#include "process_interface.h"

#include "mcoplib_ops_params_info.hpp"
#include "mcoplib_ops_params_dump.hpp"

// NV12  是yyyy....uvuvuv...  yuv420p  是 yyyyy...uuu....vvv...   3plan
//yuv2rgb+normalize+permute 

#define YUV2RGB(Y, U, V, R, G, B) \
(R) = (Y) + 1.540 * (V);\
(G) = (Y) - 0.183 * (U) - 0.459 * (V);\
(B) = (Y) + 1.816 * (U);

template<int NUM_THREADS>
__global__ void fused_preprocess_nv12_Kernel(const uint8_t* __restrict nv12Data, float* __restrict outputData, int width, int height, float scale) 
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
    const uint8_t* ptr_y = nv12Data;
    const uint8_t* ptr_uv = nv12Data + y_size;
    const uint8_t* ptr_line0_y = ptr_y + y_h * y_stride + y_w;
    const uint8_t* ptr_line1_y = ptr_line0_y + y_stride;
    const uint8_t* ptr_line_uv = ptr_uv + u_h * y_stride + y_w;
    float* ptr_r = outputData + y_h * y_stride + y_w;
    float* ptr_g = ptr_r + y_size;
    float* ptr_b = ptr_g + y_size;
    uint8_t y0[2]; 
    uint8_t y1[2];
    uint8_t uv[2];
    *(uint16_t*)y0 = *(uint16_t*)ptr_line0_y;
    *(uint16_t*)y1 = *(uint16_t*)ptr_line1_y;
    *(uint16_t*)uv = *(uint16_t*)ptr_line_uv;
    
    // 转换公式 (BT.601)
    int Y = y0[0], U = uv[0], V = uv[1];
    U = U - 128;
    V = V - 128;
    float r[2], g[2], b[2];
    YUV2RGB(Y, U, V, r[0], g[0], b[0])
    // r[0] = Y + 1.402f * (V - 128);
    // g[0] = Y - 0.34414f * (U - 128) - 0.71414f * (V - 128);
    // b[0] = Y + 1.772f * (U - 128);

    r[0] = fmaxf(0.0f, fminf(255.0f, r[0]));
    g[0] = fmaxf(0.0f, fminf(255.0f, g[0]));
    b[0] = fmaxf(0.0f, fminf(255.0f, b[0]));
    
    r[0] = r[0] * scale;
    g[0] = g[0] * scale;
    b[0] = b[0] * scale;

    Y = y0[1];
    // r[1] = Y + 1.402f * (V - 128);
    // g[1] = Y - 0.34414f * (U - 128) - 0.71414f * (V - 128);
    // b[1] = Y + 1.772f * (U - 128);
    YUV2RGB(Y, U, V, r[1], g[1], b[1])

    r[1] = fmaxf(0.0f, fminf(255.0f, r[1]));
    g[1] = fmaxf(0.0f, fminf(255.0f, g[1]));
    b[1] = fmaxf(0.0f, fminf(255.0f, b[1]));
    
    r[1] = r[1] * scale;
    g[1] = g[1] * scale;
    b[1] = b[1] * scale;
    *(float2*)ptr_r = *(float2*)r;
    *(float2*)ptr_g = *(float2*)g;
    *(float2*)ptr_b = *(float2*)b;
    Y = y1[0];
    // r[0] = Y + 1.402f * (V - 128);
    // g[0] = Y - 0.344f * (U - 128) - 0.714f * (V - 128);
    // b[0] = Y + 1.772f * (U - 128);
    YUV2RGB(Y, U, V, r[0], g[0], b[0])

    r[0] = fmaxf(0.0f, fminf(255.0f, r[0]));
    g[0] = fmaxf(0.0f, fminf(255.0f, g[0]));
    b[0] = fmaxf(0.0f, fminf(255.0f, b[0]));
    
    r[0] = r[0] * scale;
    g[0] = g[0] * scale;
    b[0] = b[0] * scale;

    Y = y1[1];
    // r[1] = Y + 1.402f * (V - 128);
    // g[1] = Y - 0.34414f * (U - 128) - 0.71414f * (V - 128);
    // b[1] = Y + 1.772f * (U - 128);
    YUV2RGB(Y,U,V, r[1], g[1], b[1])

    r[1] = fmaxf(0.0f, fminf(255.0f, r[1]));
    g[1] = fmaxf(0.0f, fminf(255.0f, g[1]));
    b[1] = fmaxf(0.0f, fminf(255.0f, b[1]));
    
    r[1] = r[1] * scale;
    g[1] = g[1] * scale;
    b[1] = b[1] * scale;
    *(float2*)(ptr_r + y_stride) = *(float2*)r;
    *(float2*)(ptr_g + y_stride) = *(float2*)g;
    *(float2*)(ptr_b + y_stride) = *(float2*)b;
}

template<int NUM_THREADS>
__global__ void fused_preprocess_yuv420_Kernel(const uint8_t* __restrict yuv420Data, float* __restrict outputData, int width, int height, float scale) 
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
    const uint8_t* ptr_y = yuv420Data;
    const uint8_t* ptr_u = ptr_y + y_size;
    const uint8_t* ptr_v = ptr_u + u_size;
    const uint8_t* ptr_line0_y = ptr_y + y_h * y_stride + y_w;
    const uint8_t* ptr_line1_y = ptr_line0_y + y_stride;
    const uint8_t* ptr_line_u = ptr_u + u_h * width + u_w;
    const uint8_t* ptr_line_v = ptr_v + u_h * width + u_w;
    float* ptr_r = outputData + y_h * y_stride + y_w;
    float* ptr_g = ptr_r + y_size;
    float* ptr_b = ptr_g + y_size;
    uint8_t y0[2]; 
    uint8_t y1[2];
    int U, V;
    *(uint16_t*)y0 = *(uint16_t*)ptr_line0_y;
    *(uint16_t*)y1 = *(uint16_t*)ptr_line1_y;

    U = *ptr_line_u;
    V = *ptr_line_v;
    
    // 转换公式 (BT.601)
    int Y = y0[0];
    float r[2], g[2], b[2];
    U = U - 128;
    V = V - 128;
    // r[0] = Y + 1.402f * (V - 128);
    // g[0] = Y - 0.34414f * (U - 128) - 0.71414f * (V - 128);
    // b[0] = Y + 1.772f * (U - 128);
    YUV2RGB(Y, U, V, r[0], g[0], b[0])

    r[0] = fmaxf(0.0f, fminf(255.0f, r[0]));
    g[0] = fmaxf(0.0f, fminf(255.0f, g[0]));
    b[0] = fmaxf(0.0f, fminf(255.0f, b[0]));
    
    r[0] = r[0] * scale;
    g[0] = g[0] * scale;
    b[0] = b[0] * scale;

    Y = y0[1];
    // r[1] = Y + 1.402f * (V - 128);
    // g[1] = Y - 0.34414f * (U - 128) - 0.71414f * (V - 128);
    // b[1] = Y + 1.772f * (U - 128);
    YUV2RGB(Y, U, V, r[1], g[1], b[1])

    r[1] = fmaxf(0.0f, fminf(255.0f, r[1]));
    g[1] = fmaxf(0.0f, fminf(255.0f, g[1]));
    b[1] = fmaxf(0.0f, fminf(255.0f, b[1]));
    
    r[1] = r[1] * scale;
    g[1] = g[1] * scale;
    b[1] = b[1] * scale;
    *(float2*)ptr_r = *(float2*)r;
    *(float2*)ptr_g = *(float2*)g;
    *(float2*)ptr_b = *(float2*)b;

    Y = y1[0];
    // r[0] = Y + 1.402f * (V - 128);
    // g[0] = Y - 0.34414f * (U - 128) - 0.71414f * (V - 128);
    // b[0] = Y + 1.772f * (U - 128);
    YUV2RGB(Y, U, V, r[0], g[0], b[0])
    r[0] = fmaxf(0.0f, fminf(255.0f, r[0]));
    g[0] = fmaxf(0.0f, fminf(255.0f, g[0]));
    b[0] = fmaxf(0.0f, fminf(255.0f, b[0]));
    
    r[0] = r[0] * scale;
    g[0] = g[0] * scale;
    b[0] = b[0] * scale;

    Y = y1[1];
    // r[1] = Y + 1.402f * (V - 128);
    // g[1] = Y - 0.34414f * (U - 128) - 0.71414f * (V - 128);
    // b[1] = Y + 1.772f * (U - 128);
    YUV2RGB(Y, U, V, r[1], g[1], b[1])
    r[1] = fmaxf(0.0f, fminf(255.0f, r[1]));
    g[1] = fmaxf(0.0f, fminf(255.0f, g[1]));
    b[1] = fmaxf(0.0f, fminf(255.0f, b[1]));
    
    r[1] = r[1] * scale;
    g[1] = g[1] * scale;
    b[1] = b[1] * scale;
    *(float2*)(ptr_r + y_stride) = *(float2*)r;
    *(float2*)(ptr_g + y_stride) = *(float2*)g;
    *(float2*)(ptr_b + y_stride) = *(float2*)b;
}

void fused_preprocess_nv12(const uint8_t* __restrict nv12Data, float* __restrict outputData, int width, int height, float scale, cudaStream_t stream)
{
    DEBUG_TRACE_PARAMS(nv12Data, outputData, width, height, scale, stream);
    DEBUG_DUMP_PARAMS(nv12Data, outputData, width, height, scale, stream);
    constexpr int num_threads = 512;
    int u_width = width / 2;
    int u_height = height / 2;
    int gridSize = (u_width * u_height + num_threads - 1) / num_threads;
    fused_preprocess_nv12_Kernel<num_threads><<<gridSize, num_threads, 0, stream>>>(nv12Data, outputData, u_width, u_height, scale);
}

void fused_preprocess_yuv420(const uint8_t* __restrict yuv420Data, float* __restrict outputData, int width, int height, float scale, cudaStream_t stream)
{
    DEBUG_TRACE_PARAMS(yuv420Data, outputData, width, height, scale, stream);
    DEBUG_DUMP_PARAMS(yuv420Data, outputData, width, height, scale, stream);
    constexpr int num_threads = 512;
    int u_width = width / 2;
    int u_height = height / 2;
    int gridSize = (u_width * u_height + num_threads - 1) / num_threads;
    fused_preprocess_yuv420_Kernel<num_threads><<<gridSize, num_threads, 0, stream>>>(yuv420Data, outputData, u_width, u_height, scale);
}