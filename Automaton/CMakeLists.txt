cmake_minimum_required(VERSION 3.15)
project(Automaton LANGUAGES CXX CUDA)

create_library(
    TARGET ${PROJECT_NAME}
    SOURCES Rule.cu Game.cu
    HEADERS Rule.cuh Game.cuh Environment.hpp
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
endif()

