#include "ConsoleRender.hpp"


namespace ca {


ConsoleRender::ConsoleRender(const ColorMap& color_map)
    : IRenderCallback()
    , _color_map(color_map)
{}

ConsoleRender::ConsoleRender(ColorMap&& color_map)
    : IRenderCallback()
    , _color_map(std::move(color_map))
{}


void ConsoleRender::operator () (int width, int height, Color* frame) const {
    for (int h = 0; h < height; ++h) {
        for (int w = 0; w < width; ++w) {
            const int index = eval_index(w, h, width, height);
            Color color = frame[index];
            printf("%c ", find_symbol(color));
        } printf("\n");
    } printf("\n");
}

int ConsoleRender::eval_index(int w_pos, int h_pos, int width, int height) const {
    if (w_pos == -1) { w_pos = width - 1; }
    if (w_pos == width) { w_pos = 0; }
    if (h_pos == -1) { h_pos = height - 1; }
    if (h_pos == height) { h_pos = 0; }

    return w_pos + (width * h_pos);
}

char ConsoleRender::find_symbol(const Color& color) const {
    auto iter = std::find_if(_color_map.cbegin(), _color_map.cend(),
        [color](const std::pair<Color, char>& element) {
            return element.first == color;
        }
    );
    return iter == _color_map.cend() ? '-' : iter->second;
}

}
