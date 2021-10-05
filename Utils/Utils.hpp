#pragma once

#include <stdio.h>
#include <cstring>
#include <utility>
#include <cassert>
#include <memory>

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

struct GLFWwindow;

namespace utils {

GLFWwindow* load_main_window(char* title, int width, int height);
bool load_opengl();

void handle_error(cudaError_t err, const char* file, int line);

#ifndef HANDLE_ERROR
#define HANDLE_ERROR(err) (utils::handle_error(err, __FILE__, __LINE__ ))
#endif // !HANDLE_ERROR


#ifndef ATTRIBS
#define ATTRIBS __host__ __device__
#endif // !ATTRIBS

#ifndef CUDA_LAUNCH
#define CUDA_LAUNCH(FUNCTION, WIDTH, HEIGHT, LIFE, FRAME) {\
    cudaEvent_t start, stop;\
    HANDLE_ERROR(cudaEventCreate(&start));\
    HANDLE_ERROR(cudaEventCreate(&stop));\
    HANDLE_ERROR(cudaEventRecord(start, 0));\
    HANDLE_ERROR(cudaEventRecord(stop, 0));\
    FUNCTION<<<WIDTH, HEIGHT>>>(LIFE, FRAME);\
    HANDLE_ERROR(cudaEventRecord(stop, 0));\
    float elapsed_time;\
    HANDLE_ERROR(cudaEventSynchronize(stop));\
    HANDLE_ERROR(cudaEventElapsedTime(&elapsed_time, start, stop));\
    printf("Elapsed time: %f\n", elapsed_time);\
    HANDLE_ERROR(cudaEventDestroy(start));\
    HANDLE_ERROR(cudaEventDestroy(stop));\
}
#endif


template <typename T>
inline T* copy_allocate_managed(T* dest, const T& source) {
    HANDLE_ERROR(cudaMallocManaged(&dest, sizeof (T)));
    *dest = source;
    return dest;
}

template <typename T>
inline T* copy_allocate_managed(const T& temp) {
    T* dest;
    HANDLE_ERROR(cudaMallocManaged(&dest, sizeof (T)));
    *dest = temp;
    return dest;
}

template <typename T> inline T* allocate_dev(size_t len) {
    T* data;
    HANDLE_ERROR(cudaMalloc(&data, len * sizeof (T)));
    return data;
}

template <typename T> inline T* allocate_managed(T* dest, size_t len) {
    assert(dest != nulptr);

    HANDLE_ERROR(cudaMallocManaged(&dest, len * sizeof (T)));
    return dest;
}

template <typename T>
inline void host_to_device(T* dest, const T* src, int len) {
    assert(dest != nullptr);
    assert(src != nullptr);

    HANDLE_ERROR(cudaMemcpy(dest, src, sizeof (T) * len,
                            cudaMemcpyHostToDevice));
}

template <typename T>
inline void device_to_device(T* dest, const T* src, int len) {
    assert(dest != nullptr);
    assert(src != nullptr);

    HANDLE_ERROR(cudaMemcpy(dest, src, sizeof (T) * len,
                            cudaMemcpyDeviceToDevice));
}

template <typename T>
inline void device_to_host(T* dest, const T* src, int len) {
    assert(dest != nullptr);
    assert(src != nullptr);

    HANDLE_ERROR(cudaMemcpy(dest, src, sizeof (T) * len,
                            cudaMemcpyDeviceToHost));
}

template <typename T>
inline void host_to_host(T* dest, const T* src, size_t len) {
    assert(dest != nullptr);
    assert(src != nullptr);
    //std::memcpy(dest, src, len);

    for (int i = 0; i < len; ++i) {
        dest[i] = src[i];
    }
}

template<typename T> T* allocate(int len, bool is_host) {
    T* out;
    if (is_host) { out = new T[len]; }
    else { out = utils::allocate_dev<T>(len); }
    return out;
}

template<typename T>
inline void copy(T* dest, const T* src, int len, bool host) {
    assert(dest != nullptr);
    assert(src != nullptr);

    if (host) { utils::host_to_host(dest, src, len); }
    else { utils::device_to_device(dest, src, len); }
}

template<typename T> inline void free(T* data, bool host) {
    if (data == nullptr) return;

    if (host) { delete[] data; }
    else { cudaFree(data); }
}

}
