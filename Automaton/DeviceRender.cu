#include <glad/glad.h>
#include <gtc/type_ptr.hpp>
#include <Utils.hpp>

#include "DeviceRender.cuh"


namespace ca {

DeviceRender::DeviceRender(Game&& game)
    : IRender()
    , _dev_game(utils::copy_allocate_managed(game))
    , _dev_frame(utils::allocate<Color>(game.len(), false))
    , _host_frame(utils::allocate<Color>(game.len(), true))
    , _width_block(game.width())
    , _height_block(game.height())
{
    _dev_game->initialize();
}

DeviceRender::DeviceRender(const Game& game)
    : IRender()
    , _dev_game(utils::copy_allocate_managed(game))
    , _dev_frame(utils::allocate<Color>(game.len(), false))
    , _host_frame(utils::allocate<Color>(game.len(), true))
    , _width_block(game.width())
    , _height_block(game.height())
{}

DeviceRender::~DeviceRender() {
    utils::free(_dev_game, false);
    utils::free(_dev_frame, false);
    utils::free(_host_frame, true);
}


__device__ Color convert(Cell cell) {
    return ((cell & 0x1) == 0x1) ? Color(1.0, 1.0, 1.0) :
                                   Color(0.0, 0.0, 0.0);
}

__global__ void __evaluate__(Game* game, Color* frame) {
    assert(game != nullptr);
    assert(frame != nullptr);

    int current_w = blockIdx.x;
    int current_h = threadIdx.x;
    int index = game->eval_index(current_w, current_h);

    game->eval_generation(current_w, current_h);
    frame[index] = convert(game->get(index));
}


void DeviceRender::evaluate() {
    cudaDeviceSynchronize();
    __evaluate__<<<_width_block, _height_block>>>(_dev_game, _dev_frame);
    cudaDeviceSynchronize();
}

void DeviceRender::render(RenderCallback callback) {
    //cudaDeviceSynchronize();
    utils::device_to_host(_host_frame, _dev_frame, _dev_game->len());
    //cudaDeviceSynchronize();

    //if (callback) {
    //    callback(_dev_game->width(), _dev_game->height(), _host_frame);
    //}
    glDrawPixels(_dev_game->width(), _dev_game->height(), GL_RGB, GL_FLOAT,
                 glm::value_ptr(*_host_frame));

    _dev_game->swap();
}

}
