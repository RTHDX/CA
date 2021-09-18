#include <glad/glad.h>
#include <gtc/type_ptr.hpp>
#include <Utils.hpp>

#include "TracedLife.cuh"


namespace game {
using namespace utils;

Life::Life(int width, int height)
    : _width(width)
    , _height(height)
    , _len(width* height)
    , _prev(allocate_dev<Cell>(_len))
    , _next(allocate_dev<Cell>(_len))
{}

Life::~Life() {
    cudaFree(_prev);
    cudaFree(_next);
}

ATTRIBS void Life::initialize(Cell* initial) {
    HANDLE_ERROR(cudaMemcpy(_prev, initial, _len * sizeof(Cell),
                 cudaMemcpyHostToDevice));
}

ATTRIBS void Life::initialize() {
    static constexpr Cell SPACE[] = {0b1, 0b0, 0b0};
    static constexpr size_t SIZE = sizeof(SPACE) / sizeof(Cell);

    Cell* initial = new Cell[_len];
    for (int i = 0; i < _len; ++i) {
        size_t index = 0 + rand() % SIZE;
        initial[i] = SPACE[index];
    }
    initialize(initial);

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

ATTRIBS Cell Life::cell_status(int w_pos, int h_pos) const {
    const int current_idx = eval_index(w_pos, h_pos);
    assert(current_idx < len());

    Cell neighbours[8];
    neighbours[0] = _prev[eval_index(w_pos - 1, h_pos - 1)];
    neighbours[1] = _prev[eval_index(w_pos, h_pos - 1)];
    neighbours[2] = _prev[eval_index(w_pos + 1, h_pos - 1)];
    neighbours[3] = _prev[eval_index(w_pos - 1, h_pos)];
    neighbours[4] = _prev[eval_index(w_pos + 1, h_pos)];
    neighbours[5] = _prev[eval_index(w_pos - 1, h_pos + 1)];
    neighbours[6] = _prev[eval_index(w_pos, h_pos + 1)];
    neighbours[7] = _prev[eval_index(w_pos + 1, h_pos + 1)];

    int count = 0;
    for (int i = 0; i < 8; ++i) { count += neighbours[i] & 0x1 == 0x1 ? 1 : 0; }

    const bool alive = _prev[current_idx] & 0x1 == 0x1 ?
                       count == 2 || count == 3 : count == 3;
    Cell cell = _prev[current_idx];
    cell = cell << 1;
    cell = cell | (alive ? 0x1 : 0x0);
    return cell;
}

ATTRIBS Cell Life::eval_cell(int w_pos, int h_pos) {
    const int current_idx = eval_index(w_pos, h_pos);
    assert(current_idx < len());

    const Cell status = cell_status(w_pos, h_pos);
    _next[current_idx] = status;
    return status;
}

ATTRIBS Cell* Life::prev() { return _prev; }
ATTRIBS Cell* Life::next() { return _next; }

ATTRIBS void Life::swap() {
    cudaDeviceSynchronize();
    HANDLE_ERROR(cudaMemcpy(_prev, _next, _len * sizeof (Cell),
                            cudaMemcpyDeviceToDevice));
    cudaDeviceSynchronize();
}

ATTRIBS int Life::len() const { return _len; }
ATTRIBS int Life::width() const { return _width; }
ATTRIBS int Life::height() const { return _height; }


ATTRIBS Color covert_to_color(Cell cell) {
    if ((cell & 0x1) == 0x1)   { return Color(1.0, 1.0, 1.0); }
    if ((cell & 0x2) == 0x2)   { return Color(0.9, 0.9, 0.9); }
    if ((cell & 0x4) == 0x4)   { return Color(0.8, 0.8, 0.8); }
    if ((cell & 0x8) == 0x8)   { return Color(0.7, 0.7, 0.7); }
    if ((cell & 0x10) == 0x10) { return Color(0.6, 0.6, 0.6); }
    if ((cell & 0x20) == 0x20) { return Color(0.5, 0.5, 0.5); }
    if ((cell & 0x30) == 0x30) { return Color(0.4, 0.4, 0.4); }
    if ((cell & 0x40) == 0x40) { return Color(0.3, 0.3, 0.3); }
    if ((cell & 0x50) == 0x50) { return Color(0.2, 0.2, 0.2); }
    if ((cell & 0x60) == 0x60) { return Color(0.1, 0.1, 0.1); }
    return Color(0.0, 0.0, 0.0);
}

__global__ static void __render__(game::Life* ctx, game::Color* frame) {
    const int current_w = blockIdx.x;
    const int current_h = threadIdx.x;
    const int current_i = ctx->eval_index(current_w, current_h);
    assert(current_i < ctx->len());

    game::Color color = covert_to_color(ctx->eval_cell(current_w, current_h));

    frame[current_i].r = color.r;
    frame[current_i].g = color.g;
    frame[current_i].b = color.b;
}


Game::Game(const Life& life)
    : _host_frame(new Color[life.len()])
    , _dev_frame(utils::allocate_dev<Color>(life.len()))
    , _device_ctx(utils::copy_allocate_managed(_device_ctx, life))
    , _block_width(life.width())
    , _thread_height(life.height())
{}

Game::~Game() {
    delete [] _host_frame;
    cudaFree(_dev_frame);
    cudaFree(_device_ctx);
}

void Game::eval_generation() {
    __render__ <<<_block_width, _thread_height>>>(_device_ctx, _dev_frame);
    _device_ctx->swap();
}

void Game::render() {
    HANDLE_ERROR(cudaMemcpy(_host_frame, _dev_frame,
                            _device_ctx->len() * sizeof(Color),
                            cudaMemcpyDeviceToHost));
    glDrawPixels(_device_ctx->width(), _device_ctx->height(), GL_RGB, GL_FLOAT,
                 glm::value_ptr(*_host_frame));
}

}
