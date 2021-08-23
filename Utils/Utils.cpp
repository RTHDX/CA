#include <stdio.h>
#include <stdlib.h>

#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "Utils.hpp"


namespace utils {

GLFWwindow* load_main_window(char* title, int width, int height) {
    if (glfwInit() == GLFW_FALSE) {
        fprintf(stderr, "[Window] Unable to load glfw\n");
        return nullptr;
    }

    GLFWwindow* window = glfwCreateWindow(width, height, title,
                                          nullptr, nullptr);

    if (window == nullptr) {
        fprintf(stderr, "[Window] Unable to create window\n");
        return nullptr;
    }

    glfwWindowHint(GLFW_SAMPLES, 4);
    glfwMakeContextCurrent(window);
    return window;
}

bool load_opengl() {
    if (!gladLoadGL()) {
        fprintf(stderr, "[Window] Unnable to load OpenGL function\n");
        return false;
    }

    int major, minor, profile_mask;
    glGetIntegerv(GL_MAJOR_VERSION, &major);
    glGetIntegerv(GL_MINOR_VERSION, &minor);
    glGetIntegerv(GL_CONTEXT_PROFILE_MASK, &profile_mask);
    printf("[Window] OpenGL contenx loaded:\n");
    printf(" - major %d\n", major);
    printf(" - minor %d\n", minor);
    printf(" - vedor %s\n", glGetString(GL_VENDOR));
    printf(" - renderer %s\n", glGetString(GL_RENDERER));
    printf(" - shading language %s\n",
        glGetString(GL_SHADING_LANGUAGE_VERSION));
    printf(" - profile mask %d\n", profile_mask);

    return true;
}

void handle_error(cudaError_t err, const char* file, int line) {
    if (err != cudaSuccess) {
        printf("%s in %s at line %d\n", cudaGetErrorString(err), file, line);
        exit(EXIT_FAILURE);
    }
}

}
