#include "IRender.hpp"


namespace ca {

class NativeRender final : public IRender {
public:
    NativeRender(Game&& game);
    NativeRender(const Game& game);
    ~NativeRender() override;

    void evaluate() override;
    void render(RenderCallback) override;

    const Color* frame() const { return _frame; }
    const Game& game() const { return _game; }

private:
    void fill_frame();
    Color convert(Cell cell) const;

private:
    Game _game;
    Color* _frame;
};

}
