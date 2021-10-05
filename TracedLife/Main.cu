#include <glad/glad.h>
#include <gtc/type_ptr.hpp>
#include <GLFW/glfw3.h>

#include <npp.h>

#include "../Automaton/Game.cuh"
#include "Utils.hpp"

int width = 1280;
int height = 952;

ca::Game create_life() {
    return ca::Game(
        ca::initialize_global("00U100000", ca::moore()),
        width, height
    );
}


ATTRIBS ca::Color convert(ca::Cell cell) {
#define APPLY_MASK(MASK) ((cell & MASK) == MASK)

    ca::Cell mask = 0b1;
    ca::Color out(1.0, 1.0, 1.0);
    for (size_t i = 0; i < 64; ++i) {
        if (APPLY_MASK(mask)) { return out; }
        out = out - ca::Color(1.0 / 64, 1.0 / 64, 1.0 / 64);
        mask = mask << 1;
    }
    return out;

#undef APPLY_MASK
}

__global__ void __evaluate__(ca::Game* game, ca::Color* frame) {
    assert(game != nullptr);
    assert(frame != nullptr);

    int current_w = blockIdx.x;
    int current_h = threadIdx.x;
    int index = game->eval_index(current_w, current_h);

    frame[index] = convert(game->eval_generation(current_w, current_h));
}

int main() {
    GLFWwindow* main_window = utils::load_main_window("Traced Life Example",
                                                      width, height);
    if (main_window == nullptr) { exit(EXIT_FAILURE); }
    if (!utils::load_opengl()) { exit(EXIT_FAILURE); }

    ca::Game life = create_life();
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

    glfwTerminate();
    exit(EXIT_SUCCESS);
}
