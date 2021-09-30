#include <gtest/gtest.h>

#include "DeviceRender.cuh"
#include "ConsoleRender.hpp"

static inline const std::vector<ca::Cell>& native_oscilator() {
    static std::vector<ca::Cell> field{
        0x0, 0x0, 0x0, 0x0, 0x0,
        0x0, 0x0, 0x1, 0x0, 0x0,
        0x0, 0x0, 0x1, 0x0, 0x0,
        0x0, 0x0, 0x1, 0x0, 0x0,
        0x0, 0x0, 0x0, 0x0, 0x0
    };
    return field;
}


testing::AssertionResult test_locality(const ca::DeviceRender& render,
                                       const std::vector<ca::Color>& colors) {
    ca::Color nord = render.frame()[render.game()->eval_index(2, 1)]
        , east = render.frame()[render.game()->eval_index(3, 2)]
        , south = render.frame()[render.game()->eval_index(2, 3)]
        , west = render.frame()[render.game()->eval_index(1, 2)]
        , center = render.frame()[render.game()->eval_index(2, 2)];

    return testing::AssertionResult(
        nord == colors[0] &&
        east == colors[1] &&
        south == colors[2] &&
        west == colors[3] &&
        center == colors[4]
    );
}


TEST(DeviceRender, life_oscilator) {
    int width = 5, height = 5;
    ca::Game game(ca::initialize_global("00U100000", ca::moore()),
                  width, height);
    game.initialize(native_oscilator().data());
    ca::DeviceRender render(game);
    const ca::Color red = ca::Color(1.0, 0.0, 0.0);
    const ca::Color blue = ca::Color(0.0, 0.0, 1.0);
    ca::ConsoleRender console_render_callback(ca::ColorMap{
        {ca::Color(1.0, 0.0, 0.0), 'r'},
        {ca::Color(0.0, 0.0, 1.0), 'b'}
    });

    render.evaluate();
    render.render(console_render_callback);
    EXPECT_TRUE(test_locality(render, {blue, red, blue, red, red}));

    render.evaluate();
    render.render(console_render_callback);
    EXPECT_TRUE(test_locality(render, {red, blue, red, blue, red}));

    render.evaluate();
    render.render(console_render_callback);
    EXPECT_TRUE(test_locality(render, {blue, red, blue, red, red}));
}
