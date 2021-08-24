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

option(CA_ENABLE_TESTS OFF)
if (${CA_ENABLE_TESTS})
    include_directories("${CMAKE_CURRENT_SOURCE_DIR}/googletest/googletest/include")
    add_subdirectory(googletest)
    enable_testing()
endif()

function (create_test_executable)
    cmake_parse_arguments(THIS "" "TARGET" "HEADERS;SOURCES;LIBS" ${ARGV})
    create_executable(
        TARGET  ${THIS_TARGET}
        SOURCES ${THIS_SOURCES}
        HEADERS ${THIS_HEADERS}
        LIBS ${THIS_LIBS} GTest::gtest GTest::gtest_main GTest::gmock
                          GTest::gmock_main
    )
endfunction (create_test_executable)

include_directories("${CMAKE_CURRENT_SOURCE_DIR}/Utils")
add_subdirectory(Utils)

include_directories("${CMAKE_CURRENT_SOURCE_DIR}/Life")
add_subdirectory(Life)

include_directories("${CMAKE_CURRENT_SOURCE_DIR}/TracedLife")
add_subdirectory(TracedLife)
