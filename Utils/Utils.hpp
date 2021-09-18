#pragma once

#include <cstring>
#include <utility>

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
    HANDLE_ERROR(cudaMallocManaged(&dest, len * sizeof (T)));
    return dest;
}

template <typename T> inline void host_to_device(T* dest, T* src, int len) {
    HANDLE_ERROR(cudaMemcpy(dest, src, sizeof (T) * len,
                            cudaMemcpyHostToDevice));
}

template <typename T> inline void device_to_device(T* dest, T* src, int len) {
    HANDLE_ERROR(cudaMemcpy(dest, src, sizeof (T) * len,
                            cudaMemcpyDeviceToDevice));
}

template <typename T> inline void device_to_host(T* dest, T* src, int len) {
    HANDLE_ERROR(cudaMemcpy(dest, src, sizeof (T) * len,
                            cudaMemcpyDeviceToHost));
}

template <typename T> inline void host_to_host(T* dest, T* src, int len) {
    std::memcpy(dest, src, len);
}

template<typename T> T* allocate(int len, bool is_host) {
    T* out;
    if (is_host) { out = new T[len]; }
    else { out = utils::allocate_dev<T>(len); }
    return out;
}

template<typename T> inline void copy(T* dest, T* src, int len, bool host) {
    if (host) { utils::host_to_host(dest, src, len); }
    else { utils::device_to_device(dest, src, len); }
}

template<typename T> inline void free(T* data, bool host) {
    if (host) { delete[] data; }
    else { cudaFree(data); }
}

}
