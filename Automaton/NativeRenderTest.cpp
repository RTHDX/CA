#include <gtest/gtest.h>

#include "NativeRender.hpp"

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

void print(const ca::Game& game, const ca::Color* frame) {
    for (int h_pos = 0; h_pos < game.height(); ++h_pos) {
        for (int w_pos = 0; w_pos < game.width(); ++w_pos) {
            ca::Color color = frame[game.eval_index(w_pos, h_pos)];
            char symbol = color == ca::Color(1.0, 0.0, 0.0) ? 'r' : 'b';
            printf("%c ", symbol);
        } printf("\n");
    } printf("\n");
}


TEST(NativeRender, life_oscillator) {
    int width = 5, height = 5;
    ca::Game game(ca::initialize_native("00U100000", ca::moore()),
                  width, height);
    game.initialize(native_oscilator().data());
    ca::NativeRender render(game, true);

    render.evaluate();
    render.render();
    print(render.game(), render.frame());
    const ca::Color red = ca::Color(1.0, 0.0, 0.0);
    const ca::Color blue = ca::Color(0.0, 0.0, 1.0);

    ca::Color nord = render.frame()[render.game().eval_index(2, 1)]
            , east = render.frame()[render.game().eval_index(3, 2)]
            , south = render.frame()[render.game().eval_index(2, 3)]
            , west = render.frame()[render.game().eval_index(1, 2)]
            , center = render.frame()[render.game().eval_index(2, 2)];
    EXPECT_EQ(nord, blue);
    EXPECT_EQ(east, red);
    EXPECT_EQ(south, blue);
    EXPECT_EQ(west, red);
    EXPECT_EQ(center, red);

    render.evaluate();
    render.render();
    print(render.game(), render.frame());
    nord = render.frame()[render.game().eval_index(2, 1)];
    east = render.frame()[render.game().eval_index(3, 2)];
    south = render.frame()[render.game().eval_index(2, 3)];
    west = render.frame()[render.game().eval_index(1, 2)];
    center = render.frame()[render.game().eval_index(2, 2)];
    EXPECT_EQ(nord, red);
    EXPECT_EQ(east, blue);
    EXPECT_EQ(south, red);
    EXPECT_EQ(west, blue);
    EXPECT_EQ(center, red);
}
