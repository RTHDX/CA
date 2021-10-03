#pragma once

#include "glm.hpp"

#include "Rule.cuh"


namespace ca {
using Color = glm::vec3;

class Game {
public:
    __host__ Game(const Rule& rule, int width, int height);
    __host__ Game(const Game&);
    __host__ Game& operator = (const Game&);
    __host__ Game(Game&&);
    __host__ Game& operator = (Game&&);
    __host__ ~Game();

    __host__ void initialize(const Cell* initial);
    __host__ void initialize();

    ATTRIBS Cell eval_generation(int w_pos, int h_pos);
    ATTRIBS void swap();

    ATTRIBS int len() const { return _width * _height; }
    ATTRIBS int width() const { return _width; }
    ATTRIBS int height() const { return _height; }
    ATTRIBS int eval_index(int w_pos, int h_pos) const;
    ATTRIBS Cell get(const int i) const;

    ATTRIBS void dump(const Cell* array) const;
    ATTRIBS Cell eval_cell(int w_pos, int h_pos) const;

    __host__ void load_prev(Cell* buffer) const;
    __host__ void load_next(Cell* buffer) const;

private:
    ATTRIBS void fill_locality(Cell* locality, int w, int h) const;

private:
    Rule _rule;
    Cell* _prev_generation = nullptr;
    Cell* _next_generation = nullptr;
    Cell* _localities = nullptr;
    int _width, _height;
};

}
