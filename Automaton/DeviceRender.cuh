#include "IRender.hpp"


namespace ca {

class DeviceRender final : public IRender {
public:
    explicit DeviceRender(Game&& game, bool silent);
    explicit DeviceRender(const Game& game, bool silent);
    ~DeviceRender() override;

    void evaluate() override;
    void render() override;

    const Color* frame() const { return _host_frame; }
    const Game* game() const { return _dev_game; }

private:
    Game* _dev_game;
    Color* _dev_frame;
    Color* _host_frame;
    bool _is_silent;
    dim3 _width_block, _height_block;
};

}
