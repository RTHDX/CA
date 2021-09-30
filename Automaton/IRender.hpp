#pragma once

#include <functional>

#include <glm.hpp>

#include "Game.cuh"


namespace ca {
using Color = glm::vec3;
using RenderCallback = std::function<void(int, int, Color*)>;

class IRender {
public:
    IRender() = default;
    virtual ~IRender() = default;

    virtual void evaluate() = 0;
    virtual void render(RenderCallback) = 0;
};


class IRenderCallback {
public:
    IRenderCallback() = default;
    virtual ~IRenderCallback() = default;

    virtual void operator () (int width, int height, Color*) const = 0;
};

}
