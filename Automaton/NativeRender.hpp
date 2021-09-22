#include "IRender.hpp"


namespace ca {

class NativeRender final : public IRender {
public:
    NativeRender(Game&& game, bool silent);
    NativeRender(const Game& game, bool silent);
    ~NativeRender() override;

    void evaluate() override;
    void render() override;

    const Color* frame() const { return _frame; }
    const Game& game() const { return _game; }

private:
    void fill_frame();
    Color convert(Cell cell) const;

private:
    Game _game;
    Color* _frame;
    bool _is_silent;
};

}
