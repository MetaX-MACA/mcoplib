#include "process_interface.h"
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
// includes CUDA Runtime
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>
#include <math.h>
#include <stdio.h>
#include <sys/time.h>

template<typename T>
void checkresult(T *origin, T*dst,int num_elements){
    int diff_nums = 0;
    for(int i = 0; i < num_elements; i++){
        if(sizeof(T) == 8) {
          if(abs((float)dst[i] - (float)origin[i]) > 0.1) {
            diff_nums++;
            if(diff_nums < 10)
            {
                printf(" aa Tv:%d,Fv:%d,index:%d\n",origin[i],dst[i],i);
            }
          }
        } else if(sizeof(T)==4){
            if(abs((float)dst[i] - (float)origin[i]) > 0.1){
                diff_nums++;
                if(diff_nums < 10)
                {
                    if(sizeof(T)==4)
                    {
                      printf(" aa Tv:%f,Fv:%f,index:%d\n",origin[i],dst[i],i);
                        // printf(" aa Tv:%.8f,Fv:%.8f,index:%d\n",origin[i],dst[i],i);
                    }else{
                        printf(" b Tv:%d,Fv:%d,index:%d\n",origin[i],dst[i],i);
                    }
                }
            }
        } else if(sizeof(T) == 1) {
            if(abs((int)dst[i] - (int)origin[i]) > 1)
            {
                diff_nums++;
                if(diff_nums < 10)
                {
                    printf("Tv:%d,Fv:%d,index:%d\n",origin[i],dst[i],i);
                }
            }
        } else {
            if(abs((float)(dst[i]) - (float)(origin[i])) > 0.0001){
                diff_nums++;
                if(diff_nums < 10)
                {
                    printf("Tv:%f,Fv:%f,index:%d\n",(float)(origin[i]),(float)(dst[i]),i);
                }
            }
        }
        
    }
    if(diff_nums > 0){
        printf("result is not right\n");
    }else{
        printf("result is right\n");
    }
}


#define CUDA_INIT() \ 
  cudaSetDevice(0); \
  cudaStream_t stream; \
  cudaStreamCreate(&stream);

#define COEF_R_Y  0.299f
#define COEF_G_Y  0.587f
#define COEF_B_Y  0.114f
#define COEF_R_U -0.14713f
#define COEF_G_U -0.28886f
#define COEF_B_U  0.436f
#define COEF_R_V  0.615f
#define COEF_G_V -0.51499f
#define COEF_B_V -0.10001f

#define RGB2Y(R, G, B, Y) \
(Y) = 0.213 * (R) + 0.715 * (G) + 0.072 * (B);

#define RGB2UV(R , G, B, U, V) \
(U) = -0.117 * (R) - 0.394 * (G) + 0.511 * (B) + 128;\
(V) = 0.511 * (R) - 0.464 * (G) - 0.047 * (B) + 128;

//V U sub 128
#define YUV2RGB(Y, U, V, R, G, B) \
(R) = (Y) + 1.540 * (V);\
(G) = (Y) - 0.183 * (U) - 0.459 * (V);\
(B) = (Y) + 1.816 * (U);

void nv12_to_rgb(const uint8_t* nv12_data, int width, int height, float scale, float* rgb_data) {
    if (!nv12_data || !rgb_data || width <= 0 || height <= 0) {
        return; // 输入参数校验
    }

    const int y_size = width * height;
    const uint8_t* y_data = nv12_data;               // Y分量起始地址
    const uint8_t* uv_data = y_data + y_size;     // UV分量起始地址
    float *ptr_r = rgb_data;
    float *ptr_g = ptr_r + y_size;
    float *ptr_b = ptr_g + y_size;
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            // 获取当前Y分量值
            int y = y_data[i * width + j];
            
            // 计算UV分量索引（每个2x2块共享一组UV）
            int uv_row = i / 2;
            int uv_col = j / 2;
            int uv_index = (uv_row * width / 2 + uv_col) * 2;
            
            // 获取U和V分量（范围需要从[0,255]转换为[-128,127]）
            int u = (int)uv_data[uv_index] - 128;
            int v = (int)uv_data[uv_index + 1] - 128;
            float r, g, b;
            YUV2RGB(y, u, v, r, g, b)
            // 范围限制（0-255）并转换为float[0.0-1.0]
            r = fmaxf(0.0f, fminf(255.0f, r));
            g = fmaxf(0.0f, fminf(255.0f, g));
            b = fmaxf(0.0f, fminf(255.0f, b));
            r = r * scale;
            g = g * scale;
            b = b * scale;
            
            ptr_r[i * width + j] = r;
            ptr_g[i * width + j] = g;
            ptr_b[i * width + j] = b;
        }
    }
}

void yuv420_to_rgb(const uint8_t* nv12_data, int width, int height, float scale, float* rgb_data) {
    if (!nv12_data || !rgb_data || width <= 0 || height <= 0) {
        return; // 输入参数校验
    }

    const int y_size = width * height;
    const uint8_t* y_data = nv12_data;               // Y分量起始地址
    const uint8_t* u_data = y_data + y_size; 
    const uint8_t* v_data = u_data + y_size / 4;   
    float *ptr_r = rgb_data;
    float *ptr_g = ptr_r + y_size;
    float *ptr_b = ptr_g + y_size;
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            // 获取当前Y分量值
            int y = y_data[i * width + j];
            
            // 计算UV分量索引（每个2x2块共享一组UV）
            int uv_row = i / 2;
            int uv_col = j / 2;
            int uv_index = (uv_row * width / 2 + uv_col);
            
            // 获取U和V分量（范围需要从[0,255]转换为[-128,127]）
            int u = (int)u_data[uv_index] - 128;
            int v = (int)v_data[uv_index] - 128;
            float r, g, b;
            YUV2RGB(y, u, v, r, g, b)

            // 确保RGB值在[0,255]范围内
            r = (r < 0) ? 0 : (r > 255) ? 255 : r;
            g = (g < 0) ? 0 : (g > 255) ? 255 : g;
            b = (b < 0) ? 0 : (b > 255) ? 255 : b;
            r = r * scale;
            g = g * scale;
            b = b * scale;
            // 存储到RGB缓冲区（RGB格式，每个像素占3字节）
            // int rgb_index = (i * width + j) * 3;
            // rgb_data[rgb_index] = (uint8_t)r;
            // rgb_data[rgb_index + 1] = (uint8_t)g;
            // rgb_data[rgb_index + 2] = (uint8_t)b;
            ptr_r[i * width + j] = r;
            ptr_g[i * width + j] = g;
            ptr_b[i * width + j] = b;
        }
    }
}

void rgb_to_nv12(const float* rgb_data, int width, int height, float scale, uint8_t* nv12_data) {
    if (!rgb_data || !nv12_data || width <= 0 || height <= 0) {
        return;
    }

    const int y_size = width * height;
    uint8_t* y_data = nv12_data;
    uint8_t* uv_data = nv12_data + y_size;
    const float* ptr_r = rgb_data;
    const float* ptr_g = ptr_r + y_size;
    const float* ptr_b = ptr_g + y_size;

    // 第一步：计算所有Y分量（与原始逻辑一致）
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            int rgb_index = (i * width + j);
            float r = ptr_r[rgb_index];
            float g = ptr_g[rgb_index];
            float b = ptr_b[rgb_index];
            uint8_t ri = (uint8_t)roundf(r * scale);
            uint8_t gi = (uint8_t)roundf(g * scale);
            uint8_t bi = (uint8_t)roundf(b * scale);

            float y;
            RGB2Y(ri, gi, bi, y);
            int yi = roundf(y);
            yi = std::max(0, std::min(255, yi));
            y_data[i * width + j] = (uint8_t)yi;
        }
    }

    // 第二步：计算UV分量（直接取偶数行、偶数列的像素，不做平均）
    for (int i = 0; i < height; i += 2) {  // 只处理偶数行（0, 2, 4...）
        for (int j = 0; j < width; j += 2) {  // 只处理偶数列（0, 2, 4...）
            // 直接使用当前偶数行、偶数列像素的RGB值计算UV
            int rgb_index = (i * width + j);
            float r = ptr_r[rgb_index];
            float g = ptr_g[rgb_index];
            float b = ptr_b[rgb_index];

            uint8_t ri = (uint8_t)roundf(r * scale);
            uint8_t gi = (uint8_t)roundf(g * scale);
            uint8_t bi = (uint8_t)roundf(b * scale);

            int u, v;
            RGB2UV(ri, gi, bi, u, v);
    
            // 钳位到[0, 255]范围
            u = (u < 0) ? 0 : (u > 255) ? 255 : u;
            v = (v < 0) ? 0 : (v > 255) ? 255 : v;

            // 计算UV在输出缓冲区中的索引
            int uv_row = i / 2;       // 偶数行映射到UV的行索引（0,1,2...）
            int uv_col = j / 2;       // 偶数列映射到UV的列索引（0,1,2...）
            int uv_index = (uv_row * (width / 2) + uv_col) * 2;

            // 存储UV分量（U在前，V在后）
            uv_data[uv_index] = (uint8_t)u;
            uv_data[uv_index + 1] = (uint8_t)v;
        }
    }
}

void test_preprocess_nv12(int width, int height){
    CUDA_INIT()
    uint8_t * yuv = nullptr;
    float* rgb = nullptr;
    float* rgb_check = nullptr;
    yuv = (uint8_t*)malloc(width*height*3/2);
    rgb = (float*)malloc(width*height*3*sizeof(float));
    rgb_check = (float*)malloc(width*height*3*sizeof(float));
    
    uint8_t* yuv_device = nullptr;
    float* rgb_device = nullptr;
    cudaMalloc((void **)&yuv_device, width*height*3/2);
    cudaMalloc((void **)&rgb_device, width*height*3*sizeof(float));
    for(int i = 0; i < width*height*3/2; i++) {
        yuv[i] = (i*7) % 128;
    }
    cudaMemcpy(yuv_device, yuv , width*height*3/2, cudaMemcpyHostToDevice);
    fused_preprocess_nv12(yuv_device, rgb_device, width, height, 1.0/255, stream);
    cudaMemcpy(rgb, rgb_device , width*height*3*sizeof(float), cudaMemcpyDeviceToHost);
    nv12_to_rgb(yuv, width, height, 1.0 / 255, rgb_check);
    checkresult<float>(rgb_check, rgb, width*height*3);
    cudaFree(yuv_device);
    cudaFree(rgb_device);
    free(yuv);
    free(rgb);
    free(rgb_check);
}

void test_prepostprocess_nv12(int width, int height){
    CUDA_INIT()
    uint8_t * yuv = nullptr;
    uint8_t* yuv_check = nullptr;
    uint8_t* yuv_check1 = nullptr;
    float* rgb = nullptr;
    float* rgb_check = nullptr;
    yuv = (uint8_t*)malloc(width*height*3/2);
    yuv_check = (uint8_t*)malloc(width*height*3/2);
    yuv_check1 = (uint8_t*)malloc(width*height*3/2);
    rgb = (float*)malloc(width*height*3*sizeof(float));
    rgb_check = (float*)malloc(width*height*3*sizeof(float));
    
    uint8_t* yuv_device = nullptr;
    uint8_t* yuv_device_check = nullptr;
    float* rgb_device = nullptr;
    cudaMalloc((void **)&yuv_device, width*height*3/2);
    cudaMalloc((void**)&yuv_device_check, width*height*3/2);
    cudaMalloc((void **)&rgb_device, width*height*3*sizeof(float));
    // for(int i = 0; i < width*height*3/2; i++) {
    //     yuv[i] = ((i*7) % 128) + 1;
    // }
    FILE * f = fopen("/home/yiyu/mcoplib/kuaishou/re_1920x1080.nv12", "rb");
    fread(yuv, 1, width*height*3/2, f);
    fclose(f);
    cudaMemcpy(yuv_device, yuv , width*height*3/2, cudaMemcpyHostToDevice);
    fused_preprocess_nv12(yuv_device, rgb_device, width, height, 1.0/255, stream);
    fused_postprocess_nv12(rgb_device, yuv_device_check, width,height,255,stream );
    cudaMemcpy(yuv_check, yuv_device_check , width*height*3/2, cudaMemcpyDeviceToHost);
    // checkresult<uint8_t>(yuv, yuv_check, width*height*3/2);

    nv12_to_rgb(yuv,width,height,1.0/255, rgb);
    cudaMemcpy(rgb_check, rgb_device , width*height*3*sizeof(float), cudaMemcpyDeviceToHost);
    checkresult<float>(rgb, rgb_check, width*height*3);
    rgb_to_nv12(rgb, width,height, 255, yuv_check1); 
    f = fopen("/home/yiyu/mcoplib/kuaishou/dst_1920x1080.nv12", "wb");
    fwrite(yuv_check1, 1, width*height*3/2, f);
    fclose(f);
    checkresult<uint8_t>(yuv_check1, yuv_check, width*height*3/2);

    fused_preprocess_yuv420(yuv_device, rgb_device, width, height, 1.0/255, stream);
    fused_postprocess_yuv420(rgb_device, yuv_device_check, width,height,255,stream );
    cudaMemcpy(yuv_check, yuv_device_check , width*height*3/2, cudaMemcpyDeviceToHost);
    checkresult<uint8_t>(yuv, yuv_check, width*height*3/2);

    cudaFree(yuv_device);
    cudaFree(yuv_device_check);
    cudaFree(rgb_device);
    free(yuv);
    free(rgb);
    free(rgb_check);
    free(yuv_check);
    free(yuv_check1);
}

void test_preprocess_yuv420(int width, int height) {
    CUDA_INIT()
    uint8_t * yuv = nullptr;
    float* rgb = nullptr, *rgb_check = nullptr;
    yuv = (uint8_t*)malloc(width*height*3/2);
    rgb = (float*)malloc(width*height*3*sizeof(float));
    rgb_check = (float*)malloc(width*height*3*sizeof(float));
    uint8_t* yuv_device = nullptr;
    float* rgb_device = nullptr;
    cudaMalloc((void **)&yuv_device, width*height*3/2);
    cudaMalloc((void **)&rgb_device, width*height*3*sizeof(float));
    for(int i = 0; i < width*height*3/2; i++) {
        yuv[i] = (i*7) % 128;
    }
    cudaMemcpy(yuv_device, yuv , width*height*3/2, cudaMemcpyHostToDevice);
    fused_preprocess_yuv420(yuv_device, rgb_device, width, height, 1.0/255, stream);
    cudaMemcpy(rgb, rgb_device , width*height*3*sizeof(float), cudaMemcpyDeviceToHost);
    yuv420_to_rgb(yuv, width, height, 1.0/255, rgb_check );
    checkresult<float>(rgb_check, rgb, width*height*3);

    cudaFree(yuv_device);
    cudaFree(rgb_device);
    free(yuv);
    free(rgb);
    free(rgb_check);

}

void test_postprocess_nv12(int width, int height)
{
    CUDA_INIT()
    uint8_t * yuv = nullptr;
    float* rgb = nullptr;
    yuv = (uint8_t*)malloc(width*height*3/2);
    rgb = (float*)malloc(width*height*3*sizeof(float));
    uint8_t* yuv_check;
    yuv_check = (uint8_t*)malloc(width*height*3/2);
    uint8_t* yuv_device = nullptr;
    float* rgb_device = nullptr;
    cudaMalloc((void **)&yuv_device, width*height*3/2);
    cudaMalloc((void **)&rgb_device, width*height*3*sizeof(float));
    for(int i = 0; i < width*height*3/2; i++) {
        rgb[i] = (i*7) % 128;
    }
    cudaMemcpy(rgb_device, rgb , width*height*3 * sizeof(float), cudaMemcpyHostToDevice);
    fused_postprocess_nv12(rgb_device, yuv_device, width, height, 255, stream);

    cudaFree(yuv_device);
    cudaFree(rgb_device);
    free(yuv);
    free(rgb);
    free(yuv_check);
}

void test_postprocess_yuv420(int width, int height) 
{
    CUDA_INIT()
    uint8_t * yuv = nullptr;
    float* rgb = nullptr;
    yuv = (uint8_t*)malloc(width*height*3/2);
    rgb = (float*)malloc(width*height*3*sizeof(float));
    uint8_t* yuv_device = nullptr;
    float* rgb_device = nullptr;
    cudaMalloc((void **)&yuv_device, width*height*3/2);
    cudaMalloc((void **)&rgb_device, width*height*3*sizeof(float));
    for(int i = 0; i < width*height*3/2; i++) {
        rgb[i] = (i*7) % 128;
    }
    cudaMemcpy(rgb_device, rgb , width*height*3 * sizeof(float), cudaMemcpyHostToDevice);
    fused_postprocess_yuv420(rgb_device, yuv_device, width, height, 255, stream);
    cudaFree(yuv_device);
    cudaFree(rgb_device);
    free(yuv);
    free(rgb);
}

int main()
{
    test_prepostprocess_nv12(1920,1080);
    // test_preprocess_nv12(1920,1080);
    // printf("test_preprocess_nv12  finished\n");
    // test_preprocess_yuv420(1920,1080);
    // printf("test_preprocess_yuv420  finished\n");
    // test_postprocess_nv12(1920, 1080);
    // printf("test_postprocess_nv12  finished\n");
    // test_postprocess_yuv420(1920, 1080);
    // printf("test_postprocess_yuv420  finished\n");
}