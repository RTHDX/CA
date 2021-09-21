#include <gtest/gtest.h>

#include "Game.cuh"


static inline const std::vector<ca::Cell>& native_oscilator() {
    static std::vector<ca::Cell> field {
        0x0, 0x0, 0x0, 0x0, 0x0,
        0x0, 0x0, 0x1, 0x0, 0x0,
        0x0, 0x0, 0x1, 0x0, 0x0,
        0x0, 0x0, 0x1, 0x0, 0x0,
        0x0, 0x0, 0x0, 0x0, 0x0
    };
    return field;
}


struct Range {
    const int w_begin, w_end;
    const int h_begin, h_end;

    int w_pos = 0, h_pos = 0;

public:
    Range(int w_begin, int w_end, int h_begin, int h_end)
        : w_begin(w_begin)
        , w_end(w_end)
        , h_begin(h_begin)
        , h_end(h_end)
    {}
};

void render(const ca::Game& game, const ca::Cell* array, Range range) {
    for (range.w_pos = range.w_begin; range.w_pos < range.w_end; ++(range.w_pos)) {
        for (range.h_pos = range.h_begin; range.h_pos < range.h_end; ++(range.h_pos)) {
            ca::Cell cell = array[game.eval_index(range.h_pos, range.w_pos)];
            char symbol = ((cell & 0x1) == 0x1) ? '1' : '0';
            printf("%c ", symbol);
        } printf("\n");
    } printf("\n");
}

void evaluate(const ca::Game& game, Range range) {
    for (range.w_pos = range.w_begin; range.w_pos < range.w_end; ++(range.w_pos)) {
        for (range.h_pos = range.h_begin; range.h_pos < range.h_end; ++(range.h_pos)) {
            game.eval_generation(range.h_pos, range.w_pos);
        }
    }
}


TEST(Game, life_oscillator_native) {
    int width = 5, height = 5;
    ASSERT_EQ(native_oscilator().size(), width * height);
    ca::Cell* temp = new ca::Cell[width * height];
    Range full(0, width, 0, height);

    ca::Rule rule = ca::initialize_native("00U100000", ca::moore());
    ca::Game game(rule, width, height);
    game.initialize(native_oscilator().data());
    game.load_prev(temp);
    render(game, temp, full);

    evaluate(game, full);
    game.load_next(temp);
    render(game, temp, full);
    game.swap();
    ca::Cell nord = temp[game.eval_index(2, 1)]
           , east = temp[game.eval_index(3, 2)]
           , south = temp[game.eval_index(2, 3)]
           , west = temp[game.eval_index(1, 2)]
           , center = temp[game.eval_index(2, 2)];

    EXPECT_EQ(nord, 0x2);
    EXPECT_EQ(east, 0x1);
    EXPECT_EQ(south, 0x2);
    EXPECT_EQ(west, 0x1);
    EXPECT_EQ(center, 0x3);

    evaluate(game, full);
    game.load_next(temp);
    render(game, temp, full);
    game.swap();

    nord = temp[game.eval_index(2, 1)];
    east = temp[game.eval_index(3, 2)];
    south = temp[game.eval_index(2, 3)];
    west = temp[game.eval_index(1, 2)];
    center = temp[game.eval_index(2, 2)];

    EXPECT_EQ(nord, 0x5);
    EXPECT_EQ(east, 0x2);
    EXPECT_EQ(south, 0x5);
    EXPECT_EQ(west, 0x2);
    EXPECT_EQ(center, 0x7);

    delete [] temp;
}


__global__ void __evaluate__(ca::Game* game) {
    int current_w = blockIdx.x;
    int current_h = threadIdx.x;
    game->eval_generation(current_w, current_h);
}

TEST(Game, life_oscillator_gpu) {
    int width = 5, height = 5;
    ca::Cell* temp = new ca::Cell[width * height];
    Range full(0, width, 0, height);

    ca::Rule rule = ca::initialize_global("00U100000", ca::moore());
    ca::Game game(rule, width, height);
    ca::Game* game_dev = utils::copy_allocate_managed(game);
    game_dev->initialize(native_oscilator().data());

    game_dev->load_prev(temp);
    render(game, temp, full);

    dim3 width_block(width), height_block(height);
    cudaDeviceSynchronize();
    __evaluate__<<<width_block, height_block>>>(game_dev);
    cudaDeviceSynchronize();
    game_dev->load_next(temp);
    render(game, temp, full);
    cudaDeviceSynchronize();
    game_dev->swap();
    cudaDeviceSynchronize();

    ca::Cell nord = temp[game.eval_index(2, 1)]
        , east = temp[game.eval_index(3, 2)]
        , south = temp[game.eval_index(2, 3)]
        , west = temp[game.eval_index(1, 2)]
        , center = temp[game.eval_index(2, 2)];

    EXPECT_EQ(nord, 0x2);
    EXPECT_EQ(east, 0x1);
    EXPECT_EQ(south, 0x2);
    EXPECT_EQ(west, 0x1);
    EXPECT_EQ(center, 0x3);

    cudaDeviceSynchronize();
    __evaluate__ << <width_block, height_block >> > (game_dev);
    cudaDeviceSynchronize();
    game_dev->load_next(temp);
    render(game, temp, full);
    cudaDeviceSynchronize();
    game_dev->swap();
    cudaDeviceSynchronize();

    nord = temp[game.eval_index(2, 1)];
    east = temp[game.eval_index(3, 2)];
    south = temp[game.eval_index(2, 3)];
    west = temp[game.eval_index(1, 2)];
    center = temp[game.eval_index(2, 2)];
    EXPECT_EQ(nord, 0x5);
    EXPECT_EQ(east, 0x2);
    EXPECT_EQ(south, 0x5);
    EXPECT_EQ(west, 0x2);
    EXPECT_EQ(center, 0x7);

    cudaDeviceSynchronize();
    __evaluate__ << <width_block, height_block >> > (game_dev);
    cudaDeviceSynchronize();
    game_dev->load_next(temp);
    render(game, temp, full);
    cudaDeviceSynchronize();
    game_dev->swap();
    cudaDeviceSynchronize();

    delete [] temp;
}
