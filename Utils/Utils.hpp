#pragma once

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


template <typename T> inline T* allocate_managed(T* dest, const T& source) {
    HANDLE_ERROR(cudaMallocManaged(&dest, sizeof (T)));
    *dest = source;
    return dest;
}

template <typename T> inline T* allocate_dev(T* dest, size_t len) {
    HANDLE_ERROR(cudaMalloc(&dest, len * sizeof (T)));
    return dest;
}

}
