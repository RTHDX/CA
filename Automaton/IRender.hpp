#include <glm.hpp>

#include "Game.cuh"


namespace ca {
using Color = glm::vec3;

class IRender {
public:
    IRender() = default;
    virtual ~IRender() = default;

    virtual void evaluate() = 0;
    virtual void render() = 0;
};

}
