#include <Utils.hpp>

#include "Game.cuh"


namespace ca {

__host__ Game::Game(const Rule& rule, int width, int height)
    : _rule(rule)
    , _prev_generation(utils::allocate_dev<Cell>(width * height))
    , _next_generation(utils::allocate_dev<Cell>(width * height))
    , _width(width)
    , _height(height)
{}

__host__ Game::Game(const Game& old) {
    int len = old._width * old._height;
    _rule = old._rule;
    _width = old._width;
    _height = old._height;

    _prev_generation = utils::allocate_dev<Cell>(len);
    utils::copy(_prev_generation, old._prev_generation, len, false);
    _next_generation = utils::allocate_dev<Cell>(_width * _height);
    utils::copy(_prev_generation, old._prev_generation, len, false);
}

__host__ Game& Game::operator = (const Game& old) {
    if (this == &old) return *this;

    int len = old._width * old._height;
    _rule = old._rule;
    _width = old._width;
    _height = old._height;

    Cell* temp_prev = utils::allocate<Cell>(len, false);
    utils::copy(temp_prev, old._prev_generation, len, false);
    utils::free(_prev_generation, false);
    _prev_generation = temp_prev;

    Cell* temp_next = utils::allocate<Cell>(len, false);
    utils::copy(temp_next, old._next_generation, len, false);
    utils::free(_next_generation, false);
    _next_generation = temp_next;

    return *this;
}

__host__ Game::Game(Game&& old) {
    _rule = old._rule;
    _width = old._width;
    _height = old._height;
    _prev_generation = old._prev_generation;
    _next_generation = old._next_generation;
}

__host__ Game& Game::operator = (Game&& old) {
    if (this == &old) return *this;

    utils::free(_prev_generation, false);
    utils::free(_next_generation, false);
    _prev_generation = old._prev_generation;
    _next_generation = old._next_generation;
    _rule = old._rule;
    _width = old._width;
    _height = old._height;

    return *this;
}

__host__ Game::~Game() {
    utils::free(_prev_generation, false);
    utils::free(_next_generation, false);
}

ATTRIBS void Game::initialize(Cell* initial) {
    HANDLE_ERROR(cudaMemcpy(_prev_generation, initial, len() * sizeof(Cell),
                            cudaMemcpyHostToDevice));
}

ATTRIBS void Game::initialize() {
    static constexpr Cell SPACE[] = {0x1, 0x0, 0x0};
    static constexpr size_t SIZE = sizeof(SPACE) / sizeof(Cell);

    Cell* initial = new Cell[len()];
    for (int i = 0; i < len(); ++i) {
        size_t index = 0 + rand() % SIZE;
        initial[i] = SPACE[index];
    }
    initialize(initial);

    delete[] initial;
}

ATTRIBS void Game::eval_generation(int w_pos, int h_pos) const {
    _next_generation[eval_index(w_pos, h_pos)] = eval_cell(w_pos, h_pos);
}

ATTRIBS void Game::swap() {
    utils::device_to_device(_prev_generation, _next_generation, len());
}

ATTRIBS int Game::eval_index(int w_pos, int h_pos) const {
    if (w_pos == -1) { w_pos = width() - 1; }
    if (w_pos == width()) { w_pos = 0; }
    if (h_pos == -1) { h_pos = height() - 1; }
    if (h_pos == height()) { h_pos = 0; }

    int index = w_pos + (width() * h_pos);
    assert(index < len());
    return index;
}

ATTRIBS Cell Game::eval_cell(int w_pos, int h_pos) const {
    Cell* locality = new Cell[_rule.len()];
    fill_locality(locality, w_pos, h_pos);

    const int current_index = eval_index(w_pos, h_pos);
    assert(current_index < len());
    Cell result = _rule.apply(locality, _prev_generation[current_index]);

    delete [] locality;
    return result;
}

ATTRIBS void Game::fill_locality(Cell* locality, int w_pos, int h_pos) const {
#define APPLY_MASK(MASK) ((_rule.env() & (MASK)) == (MASK))

    int index = 0;
    if (APPLY_MASK(Environment::NORD_WEST)) {
        locality[index] = _prev_generation[eval_index(w_pos - 1, h_pos - 1)];
        index++;
    }
    if (APPLY_MASK(Environment::NORD)) {
        locality[index] = _prev_generation[eval_index(w_pos, h_pos - 1)];
        index++;
    }
    if (APPLY_MASK(Environment::NORD_EAST)) {
        locality[index] = _prev_generation[eval_index(w_pos + 1, h_pos - 1)];
        index++;
    }
    if (APPLY_MASK(Environment::EAST)) {
        locality[index] = _prev_generation[eval_index(w_pos + 1, h_pos)];
        index++;
    }
    if (APPLY_MASK(Environment::SOUTH_EAST)) {
        locality[index] = _prev_generation[eval_index(w_pos + 1, h_pos + 1)];
        index++;
    }
    if (APPLY_MASK(Environment::SOUTH)) {
        locality[index] = _prev_generation[eval_index(w_pos, h_pos + 1)];
        index++;
    }
    if (APPLY_MASK(Environment::SOUTH_WEST)) {
        locality[index] = _prev_generation[eval_index(w_pos - 1, h_pos + 1)];
        index++;
    }
    if (APPLY_MASK(Environment::WEST)) {
        locality[index] = _prev_generation[eval_index(w_pos - 1, h_pos)];
        index++;
    }
    if (APPLY_MASK(Environment::CENTER)) {
        locality[index] = _prev_generation[eval_index(w_pos, h_pos)];
        index++;
    }

#undef APPLY_MASK
}

}
