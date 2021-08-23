#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <glm.hpp>

namespace game {

using Color = glm::vec3;
using Cell = bool;

class Life {
public:
    Life(int width, int height);
    ~Life();

    ATTRIBS void initialize();
    ATTRIBS int eval_index(int w_pos, int h_pos) const;
    ATTRIBS bool cell_status(int w_pos, int h_pos) const;
    ATTRIBS bool eval_generation(int w_pos, int h_pos);

    ATTRIBS bool* prev();
    ATTRIBS bool* next();

    ATTRIBS int len() const;
    ATTRIBS int width() const;
    ATTRIBS int height() const;

private:
    int _width, _height, _len;
    Cell* _prev_device_generation;
    Cell* _next_device_generation;
};


class Game {
public:
    Game(const Life& life);
    ~Game();

    void render();

private:
    Color* _host_frame = nullptr;
    Color* _dev_frame = nullptr;
    Life* _device_ctx = nullptr;
    dim3 _block_width, _thread_height;
};

}
