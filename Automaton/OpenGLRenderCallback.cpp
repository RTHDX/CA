#include <glad/glad.h>
#include <gtc/type_ptr.hpp>

#include "OpenGLRenderCallback.hpp"


namespace ca {

void OpenGLRenderCallback::operator () (int width, int height, Color* frame)
                                        const {
    glDrawPixels(width, height, GL_RGB, GL_FLOAT, glm::value_ptr(*frame));
}

}
