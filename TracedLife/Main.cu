#include <GLFW/glfw3.h>
#include <Utils.hpp>

#include "TracedLife.cuh"


int main() {
    int width = 1280, height = 860;
    GLFWwindow* window = utils::load_main_window("Life", width, height);
    if (!utils::load_opengl()) { exit(EXIT_FAILURE); }

    game::Life life(width, height);
    life.initialize();
    game::Game game(life);

    while (!glfwWindowShouldClose(window)) {
        glfwSwapBuffers(window);
        game.render();
        glfwPollEvents();
    }

    exit(EXIT_SUCCESS);
}

