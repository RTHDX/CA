#pragma once

#include "Rule.cuh"


namespace ca {

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

    ATTRIBS void eval_generation(int w_pos, int h_pos) const;
    ATTRIBS void swap();

    ATTRIBS Cell* generation() const { return _next_generation; }
    ATTRIBS int len() const { return _width * _height; }
    ATTRIBS int width() const { return _width; }
    ATTRIBS int height() const { return _height; }
    ATTRIBS int eval_index(int w_pos, int h_pos) const;

    __host__ void load_prev(Cell* buffer) const;
    __host__ void load_next(Cell* buffer) const;

private:
    ATTRIBS Cell eval_cell(int w_pos, int h_pos) const;
    ATTRIBS void fill_locality(Cell* locality, int w, int h) const;
    ATTRIBS void dump(const Cell* array) const;

private:
    Rule _rule;
    Cell* _prev_generation;
    Cell* _next_generation;
    int _width, _height;
};


class HostGame {

};

}
