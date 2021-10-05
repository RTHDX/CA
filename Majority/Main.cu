#include <glad/glad.h>
#include <gtc/type_ptr.hpp>
#include <GLFW/glfw3.h>

#include "../Automaton/Game.cuh"
#include "Utils.hpp"

int width = 1280;
int height = 860;

ca::Game create_majority() {
    return ca::Game(
        ca::initialize_global("0000011111", ca::full()),
        width, height
    );
}


__device__ ca::Color convert(ca::Cell cell) {
    return ((cell & 0x1) == 0x1) ? ca::Color(1.0, 1.0, 1.0) :
                                   ca::Color(0.0, 0.0, 0.0);
}

__global__ void __evaluate__(ca::Game* game, ca::Color* frame) {
    assert(game != nullptr);
    assert(frame != nullptr);

    int current_w = blockIdx.x;
    int current_h = threadIdx.x;
    int index = game->eval_index(current_w, current_h);

    ca::Cell current = game->eval_generation(current_w, current_h);
    frame[index] = convert(current);
}


int main() {
    GLFWwindow* main_window = utils::load_main_window("Majority example",
                                                      width, height);
    if (main_window == nullptr) { exit(EXIT_FAILURE); }
    if (!utils::load_opengl()) { exit(EXIT_FAILURE); }

    ca::Game life = create_majority();
    ca::Game* dev_life = utils::copy_allocate_managed(life);
    ca::Color* dev_frame = utils::allocate<ca::Color>(width * height, false);
    ca::Color* host_frame = utils::allocate<ca::Color>(width * height, true);
    dim3 width_block(width), height_block(height);
    dev_life->initialize();

    while (!glfwWindowShouldClose(main_window)) {
        glfwSwapBuffers(main_window);

        CUDA_LAUNCH(__evaluate__, width_block, height_block, dev_life,
                    dev_frame)
        utils::device_to_host(host_frame, dev_frame, width * height);
        glDrawPixels(width, height, GL_RGB, GL_FLOAT,
                     glm::value_ptr(*host_frame));
        dev_life->swap();

        glfwPollEvents();
    }

    utils::free(dev_life, false);
    utils::free(dev_frame, false);
    utils::free(host_frame, true);

    exit(EXIT_SUCCESS);
}
