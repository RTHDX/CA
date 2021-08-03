#include <stdio.h>
#include <stdlib.h>
#include <memory.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <glm.hpp>
#include <gtc/type_ptr.hpp>

#include "Utils.cuh"
#include "Game.cuh"


__global__ void __render__(game::Life* ctx, game::Color* frame) {
    const int current_w = blockIdx.x;
    const int current_h = threadIdx.x;
    const int current_i = ctx->eval_index(current_w, current_h);
    assert(current_i < ctx->len());

    game::Color color = ctx->eval_generation(current_w, current_h) ?
                        game::Color(0.0f, 1.0f, 0.0f) :
                        game::Color(0.1f, 0.1f, 0.1f);

    frame[current_i].r = color.r;
    frame[current_i].g = color.g;
    frame[current_i].b = color.b;
}

__global__ void __swap__(game::Life* ctx) {
    const int current_w = blockIdx.x;
    const int current_h = threadIdx.x;
    const int current_i = ctx->eval_index(current_w, current_h);
    assert(current_i < ctx->len());

    ctx->prev()[current_i] = ctx->next()[current_i];
}

void render(game::Life* ctx, const dim3& width, const dim3& height,
            game::Color* frame) {
    cudaDeviceSynchronize();
    __render__<<<width, height>>>(ctx, frame);
    cudaDeviceSynchronize();
    __swap__<<<width, height>>>(ctx);
    cudaDeviceSynchronize();
}


int main() {
    int width = 1280, height = 860;
    GLFWwindow* window = utils::load_main_window("Life", width, height);
    if (!utils::load_opengl()) { exit(EXIT_FAILURE); }

    game::Life life(width, height);
    life.initialize();
    int len = life.len();
    game::Life* dev_ptr = nullptr;
    HANDLE_ERROR(cudaMallocManaged(&dev_ptr, sizeof (game::Life)));
    *dev_ptr = life;
    game::Color* frame = new game::Color [len];
    game::Color* dev_frame = nullptr;
    HANDLE_ERROR(cudaMallocManaged(&dev_frame, len * sizeof (game::Color)));

    dim3 block_width(width), thread_height(height);
    while (!glfwWindowShouldClose(window)) {
        glfwSwapBuffers(window);
        render(dev_ptr, block_width, thread_height, dev_frame);
        HANDLE_ERROR(cudaMemcpy(frame, dev_frame, len * sizeof (glm::vec3),
                     cudaMemcpyDeviceToHost));
        glDrawPixels(width, height, GL_RGB, GL_FLOAT, glm::value_ptr(*frame));
        glfwPollEvents();
    }

    delete [] frame;
    cudaFree(dev_ptr);
    cudaFree(dev_frame);
    exit(EXIT_SUCCESS);
}

