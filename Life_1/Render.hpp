#include <glm/glm.hpp>

#include <Automaton/Game.cuh>

using Color = glm::vec3;


namespace life {

class Render {
public:
    Render(ca::Game& game);
    ~Render();

    void run();

private:
    ca::Game* _global_game;
    dim3 _width, dim3 _height;
    Color* _colors;
};

}
