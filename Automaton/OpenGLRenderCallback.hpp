#pragma once

#include "IRender.hpp"


namespace ca {

class OpenGLRenderCallback final : public IRenderCallback {
public:
    OpenGLRenderCallback() = default;
    ~OpenGLRenderCallback() override = default;

    void operator () (int width, int height, Color* frame) const override;
};

}
