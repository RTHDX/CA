#include <Utils/Utils.hpp>

#include "Render.hpp"


namespace life {

Render::Render(ca::Game& game)
    : _global_game(utils::copy_allocate_managed(game))
    , _width(game.width())
    , _height(game.height())
    , _
{}

Render::~Render() {
    utils::free(_global_game, false);
}


__global__ void render(ca::Game* game, )


void Render::run() {

}

}
