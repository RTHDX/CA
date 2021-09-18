#include <gtest/gtest.h>

#include "Game.cuh"


TEST(Game, life_oscillator) {
    ca::Cell field[] = {
        0x0, 0x0, 0x0, 0x0, 0x0,
        0x0, 0x0, 0x1, 0x0, 0x0,
        0x0, 0x0, 0x1, 0x0, 0x0,
        0x0, 0x0, 0x1, 0x0, 0x0,
        0x0, 0x0, 0x0, 0x0, 0x0
    };

    ca::Rule rule = ca::initialize_native("0U100000", ca::moore());
    ca::Game* dev_game = utils::copy_allocate_managed(ca::Game(rule, 5, 5));

    cudaFree(dev_game);
}
