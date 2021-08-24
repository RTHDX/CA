#include <gtest/gtest.h>

#include "TracedLife.cuh"


class GameTest : public testing::Test {
public:
    GameTest();
    ~GameTest();

    void eval_generation();

public:
    int _width = 5, _height = 5;
    game::Cell* _generation = nullptr;
    game::Game* _game = nullptr;
    game::Life* _life = nullptr;
};
