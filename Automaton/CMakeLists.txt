cmake_minimum_required(VERSION 3.15)
project(Automaton LANGUAGES CXX CUDA)

create_library(
    TARGET ${PROJECT_NAME}
    SOURCES Rule.cu  Game.cu NativeRender.cpp DeviceRender.cu
    HEADERS Rule.cuh Game.cuh Environment.hpp IRender.hpp NativeRender.hpp
            DeviceRender.cuh
    LIBS Utils
)

if (${CA_ENABLE_TESTS})
    create_test_executable(
        TARGET RuleTest
        SOURCES RuleTest.cu
        HEADERS
        LIBS ${PROJECT_NAME}
    )

    create_test_executable(
        TARGET GameTest
        SOURCES GameTest.cu
        HEADERS
        LIBS ${PROJECT_NAME}
    )

    create_test_executable(
        TARGET NativeRenderTest
        SOURCES NativeRenderTest.cpp
        HEADERS
        LIBS ${PROJECT_NAME}
    )

    create_test_executable(
        TARGET DeviceRenderTest
        SOURCES DeviceRenderTest.cu
        HEADERS
        LIBS ${PROJECT_NAME}
    )
endif()

