#include <glad/glad.h>
#include <gtc/type_ptr.hpp>

#include <Utils.hpp>

#include "NativeRender.hpp"


namespace ca {

NativeRender::NativeRender(Game&& game, bool silent)
    : IRender()
    , _game(std::move(game))
    , _frame(utils::allocate<Color>(game.len(), true))
    , _is_silent(silent) {
    _game.initialize();
}

NativeRender::NativeRender(const Game& game, bool silent)
    : IRender()
    , _game(game)
    , _frame(utils::allocate<Color>(game.len(), true))
    , _is_silent(silent)
{}

NativeRender::~NativeRender() {
    utils::free(_frame, true);
}

void NativeRender::evaluate() {
    for (int h_pos = 0; h_pos < _game.height(); ++h_pos) {
        for (int w_pos = 0; w_pos < _game.width(); ++w_pos) {
            _game.eval_generation(w_pos, h_pos);
        }
    }
}

void NativeRender::render() {
    fill_frame();

    if (!_is_silent) {
        glDrawPixels(_game.width(), _game.height(), GL_RGB, GL_FLOAT,
                     glm::value_ptr(*_frame));
    }

    _game.swap();
}

void NativeRender::fill_frame() {
    for (size_t index = 0; index < _game.len(); ++index) {
        _frame[index] = convert(_game.get(index));
    }
}

Color NativeRender::convert(Cell cell) const {
    return ((cell & 0x1) == 0x1) ? Color(1.0, 0.0, 0.0) :
                                   Color(0.0, 0.0, 1.0);
}

}
