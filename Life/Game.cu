#include <glad/glad.h>
#include <gtc/type_ptr.hpp>
#include <Utils.hpp>

#include "Game.cuh"


namespace game {

Life::Life(int width, int height)
    : _width(width)
    , _height(height)
    , _len(width* height) {
    HANDLE_ERROR(cudaMalloc(&_next_device_generation, _len * sizeof(bool)));
    HANDLE_ERROR(cudaMalloc(&_prev_device_generation, _len * sizeof(bool)));
}

Life::~Life() {
    cudaFree(_prev_device_generation);
    cudaFree(_next_device_generation);
}

ATTRIBS void Life::initialize() {
    static constexpr bool SPACE[] = { true, false, false };
    static constexpr size_t SIZE = sizeof(SPACE) / sizeof(bool);

    bool* initial = new bool[_len];
    for (int i = 0; i < _len; ++i) {
        bool index = 0 + rand() % SIZE;
        initial[i] = SPACE[index];
    }
    HANDLE_ERROR(cudaMemcpy(_prev_device_generation, initial,
        _len * sizeof(bool), cudaMemcpyHostToDevice));

    delete[] initial;
}

ATTRIBS int Life::eval_index(int w_pos, int h_pos) const {
    if (w_pos == -1) { w_pos = width() - 1; }
    if (w_pos == width()) { w_pos = 0; }
    if (h_pos == -1) { h_pos = height() - 1; }
    if (h_pos == height()) { h_pos = 0; }

    int index = w_pos + (width() * h_pos);
    assert(index < len());
    return index;
}

ATTRIBS bool Life::cell_status(int w_pos, int h_pos) const {
    const int current_idx = eval_index(w_pos, h_pos);
    assert(current_idx < len());

    bool neighbours[8];
    neighbours[0] = _prev_device_generation[eval_index(w_pos - 1, h_pos - 1)];
    neighbours[1] = _prev_device_generation[eval_index(w_pos, h_pos - 1)];
    neighbours[2] = _prev_device_generation[eval_index(w_pos + 1, h_pos - 1)];
    neighbours[3] = _prev_device_generation[eval_index(w_pos - 1, h_pos)];
    neighbours[4] = _prev_device_generation[eval_index(w_pos + 1, h_pos)];
    neighbours[5] = _prev_device_generation[eval_index(w_pos - 1, h_pos + 1)];
    neighbours[6] = _prev_device_generation[eval_index(w_pos, h_pos + 1)];
    neighbours[7] = _prev_device_generation[eval_index(w_pos + 1, h_pos + 1)];

    int count = 0;
    for (int i = 0; i < 8; ++i) { count += neighbours[i] ? 1 : 0; }

    return _prev_device_generation[current_idx] ?
        count == 2 || count == 3 :
        count == 3;
}

ATTRIBS bool Life::eval_generation(int w_pos, int h_pos) {
    const int current_idx = eval_index(w_pos, h_pos);
    assert(current_idx < len());

    const bool status = cell_status(w_pos, h_pos);
    _next_device_generation[current_idx] = status;
    return status;
}

ATTRIBS bool* Life::prev() { return _prev_device_generation; }
ATTRIBS bool* Life::next() { return _next_device_generation; }

ATTRIBS int Life::len() const { return _len; }
ATTRIBS int Life::width() const { return _width; }
ATTRIBS int Life::height() const { return _height; }


__global__ static void __render__(game::Life* ctx, game::Color* frame) {
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

__global__ static void __swap__(game::Life* ctx) {
    const int current_w = blockIdx.x;
    const int current_h = threadIdx.x;
    const int current_i = ctx->eval_index(current_w, current_h);
    assert(current_i < ctx->len());

    ctx->prev()[current_i] = ctx->next()[current_i];
}


Game::Game(const Life& life)
    : _host_frame(new Color[life.len()])
    , _dev_frame(utils::allocate_dev(_dev_frame, life.len()))
    , _device_ctx(utils::allocate_managed(_device_ctx, life))
    , _block_width(life.width())
    , _thread_height(life.height())
{}

Game::~Game() {
    delete [] _host_frame;
    cudaFree(_dev_frame);
    cudaFree(_device_ctx);
}

void Game::render() {
    cudaDeviceSynchronize();
    __render__<<<_block_width, _thread_height>>>(_device_ctx, _dev_frame);
    cudaDeviceSynchronize();
    __swap__<<<_block_width, _thread_height>>>(_device_ctx);
    cudaDeviceSynchronize();
    HANDLE_ERROR(cudaMemcpy(_host_frame, _dev_frame,
                            _device_ctx->len() * sizeof(Color),
                            cudaMemcpyDeviceToHost));
    glDrawPixels(_device_ctx->width(), _device_ctx->height(), GL_RGB, GL_FLOAT,
                 glm::value_ptr(*_host_frame));
}

}
