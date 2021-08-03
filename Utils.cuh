#pragma once

struct GLFWwindow;

namespace utils {

GLFWwindow* load_main_window(char* title, int width, int height);
bool load_opengl();

void handle_error(cudaError_t err, const char* file, int line);
#define HANDLE_ERROR(err) (utils::handle_error(err, __FILE__, __LINE__ ))

}
