#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <glm.hpp>

#include <Utils.hpp>


namespace game {
using Color = glm::vec3;
using Cell = int;

class Life {
public:
    Life(int width, int height);
    ~Life();

    ATTRIBS void initialize(Cell* initial);
    ATTRIBS void initialize();
    ATTRIBS int eval_index(int w_pos, int h_pos) const;
    ATTRIBS Cell cell_status(int w_pos, int h_pos) const;
    ATTRIBS Cell eval_cell(int w_pos, int h_pos);

    ATTRIBS Cell* prev();
    ATTRIBS Cell* next();
    ATTRIBS void swap();

    ATTRIBS int len() const;
    ATTRIBS int width() const;
    ATTRIBS int height() const;

private:
    int _width, _height, _len;
    Cell* _prev;
    Cell* _next;
};

ATTRIBS Color covert_to_color(Cell cell);

class Game {
public:
    Game(const Life& life);
    ~Game();

    void eval_generation();
    void render();
    Life* ctx() { return _device_ctx; }

private:
    Color* _host_frame = nullptr;
    Color* _dev_frame = nullptr;
    Life* _device_ctx = nullptr;
    dim3 _block_width, _thread_height;
};

}
