#pragma once

#include <map>
#include <vector>

#include "IRender.hpp"


namespace ca {
using ColorMap = std::vector<std::pair<Color, char>>;

class ConsoleRender final : public IRenderCallback {
public:
    explicit ConsoleRender(const ColorMap& color_map);
    explicit ConsoleRender(ColorMap&& color_map);
    ~ConsoleRender() override = default;

    void operator () (int width, int height, Color* frame) const override;

private:
    int eval_index(int w_pos, int h_pos, int width, int height) const;
    char find_symbol(const Color& color) const;

private:
    ColorMap _color_map;
};

}
