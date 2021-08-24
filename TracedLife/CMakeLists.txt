cmake_minimum_required(VERSION 3.15)
project(TracedLife LANGUAGES CXX CUDA)

create_library(
    TARGET impl
    SOURCES TracedLife.cu
    HEADERS TracedLife.cuh
)

create_executable(
    TARGET ${PROJECT_NAME}
    SOURCES Main.cu
    HEADERS
    LIBS Utils impl
)

if (${CA_ENABLE_TESTS})
    create_test_executable(
        TARGET "${PROJECT_NAME}_Test"
        SOURCES UnitTest.cu TracedLife.cu
        HEADERS TracedLife.cuh
        LIBS Utils
    )
endif()
