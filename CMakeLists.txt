cmake_minimum_required(VERSION 3.15)
project(Cellular_Automaton LANGUAGES CXX CUDA)


function (create_executable)
    cmake_parse_arguments(THIS "" "TARGET" "HEADERS;SOURCES;LIBS" ${ARGV})
    message("Preparing executable: ${THIS_TARGET}")
    add_executable(${THIS_TARGET} ${THIS_HEADERS} ${THIS_SOURCES})
    target_include_directories(${THIS_TARGET} PRIVATE
        ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
    )
    target_link_libraries(${THIS_TARGET} PRIVATE
        ${OPENGL_LIBRARIES} glfw glad ${THIS_LIBS}
    )
    set_property(TARGET ${THIS_TARGET} PROPERTY CXX_STANDARD 17)
    set_property(TARGET ${THIS_TARGET} PROPERTY CUDA_SEPARABLE_COMPILATION ON)
endfunction (create_executable)

function (create_library)
    cmake_parse_arguments(THIS "" "TARGET" "HEADERS;SOURCES;LIBS" ${ARGV})
    message("Preparing library: ${THIS_TARGET}")
    add_library(${THIS_TARGET} STATIC ${THIS_HEADERS} ${THIS_SOURCES})
    target_include_directories(${THIS_TARGET} PRIVATE
        ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
    )
    target_link_libraries(${THIS_TARGET} PRIVATE
        ${OPENGL_LIBRARIES} glfw glad ${THIS_LIBS}
    )
    set_property(TARGET ${THIS_TARGET} PROPERTY CXX_STANDARD 17)
    set_property(TARGET ${THIS_TARGET} PROPERTY CUDA_SEPARABLE_COMPILATION ON)
endfunction (create_library)

include_directories("${CMAKE_CURRENT_SOURCE_DIR}/glad/include")
add_subdirectory(glad)

include_directories("${CMAKE_CURRENT_SOURCE_DIR}/glfw/include")
add_subdirectory(glfw)

include_directories("${CMAKE_CURRENT_SOURCE_DIR}/glm/glm")
add_subdirectory(glm)


create_library(
    TARGET Utils
    SOURCES Utils.cu
    HEADERS Utils.cuh
)

create_executable(
    TARGET ${PROJECT_NAME}
    SOURCES Main.cu
    HEADERS
    LIBS Utils
)
