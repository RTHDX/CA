#include "UnitTest.cuh"


using namespace testing;
using namespace game;

inline void copy_to_host(Cell* dst, Cell* src, size_t len) {
    HANDLE_ERROR(cudaMemcpy(dst, src, len * sizeof (Cell),
                            cudaMemcpyDeviceToHost
        )
    );
}

inline void render(Cell* array, const Life& life) {
    for (size_t i = 0; i < life.width(); ++i) {
        for (size_t j = 0; j < life.height(); ++j) {
            const int index = life.eval_index(i, j);
            Cell cell = array[index];
            std::cout << (cell & 0x1 == 0x1 ? '1': '0') << ' ';
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

static Cell* pulsar() {
    static Cell _pulsar[] = {
        0, 0, 0, 0, 0,
        0, 0, 1, 0, 0,
        0, 0, 1, 0, 0,
        0, 0, 1, 0, 0,
        0, 0, 0, 0, 0
    };
    return _pulsar;
}


GameTest::GameTest()
    : testing::Test()
    , _width(5)
    , _height(5)
    , _generation(new Cell[_width * _height])
{
    _life = new Life(_width, _height);
    _life->initialize(pulsar());
    _game = new Game(*_life);
}

GameTest::~GameTest() {
    delete _game;
    delete _life;
    delete [] _generation;
}

void GameTest::eval_generation() {
    _game->eval_generation();
    copy_to_host(_generation, _game->ctx()->prev(), _life->len());
    render(_generation, *_life);
}

TEST_F(GameTest, generations) {
    int w_mid = _life->width() / 2, h_mid = _life->height() / 2;

    eval_generation();
    EXPECT_EQ(_generation[_life->eval_index(w_mid, h_mid)], 3);
    EXPECT_EQ(_generation[_life->eval_index(w_mid - 1, h_mid)], 1);
    EXPECT_EQ(_generation[_life->eval_index(w_mid + 1, h_mid)], 1);

    eval_generation();
    EXPECT_EQ(_generation[_life->eval_index(w_mid, h_mid)], 7);
    EXPECT_EQ(_generation[_life->eval_index(w_mid - 1, h_mid)], 2);
    EXPECT_EQ(_generation[_life->eval_index(w_mid + 1, h_mid)], 2);

    eval_generation();
    EXPECT_EQ(_generation[_life->eval_index(w_mid, h_mid)], 15);
    EXPECT_EQ(_generation[_life->eval_index(w_mid - 1, h_mid)], 5);
    EXPECT_EQ(_generation[_life->eval_index(w_mid + 1, h_mid)], 5);
}

TEST_F(GameTest, color_converter) {
    int w_mid = _life->width() / 2, h_mid = _life->height() / 2;

    eval_generation();
    eval_generation();
    EXPECT_EQ(_generation[_life->eval_index(w_mid, h_mid)], 7);
    EXPECT_EQ(covert_to_color(_generation[_life->eval_index(w_mid, h_mid)]),
              Color(1.0, 0.0, 0.0));
    EXPECT_EQ(_generation[_life->eval_index(w_mid - 1, h_mid)], 2);
    EXPECT_EQ(covert_to_color(_generation[_life->eval_index(w_mid - 1, h_mid)]),
              Color(0.0, 0.0, 1.0));
    EXPECT_EQ(_generation[_life->eval_index(w_mid + 1, h_mid)], 2);
    EXPECT_EQ(covert_to_color(_generation[_life->eval_index(w_mid + 1, h_mid)]),
              Color(0.0, 0.0, 1.0));
}


int main(int argc, char** argv) {
    InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
