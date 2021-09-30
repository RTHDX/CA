#include "IRender.hpp"


namespace ca {

class DeviceRender final : public IRender {
public:
    explicit DeviceRender(Game&& game);
    explicit DeviceRender(const Game& game);
    ~DeviceRender() override;

    void evaluate() override;
    void render(RenderCallback) override;

    const Color* frame() const { return _host_frame; }
    const Game* game() const { return _dev_game; }

private:
    Game* _dev_game;
    Color* _dev_frame;
    Color* _host_frame;
    dim3 _width_block, _height_block;
};

}
