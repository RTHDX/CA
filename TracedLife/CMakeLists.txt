cmake_minimum_required(VERSION 3.15)
project(TracedLife LANGUAGES CXX CUDA)

create_executable(
    TARGET ${PROJECT_NAME}
    SOURCES Main.cu Game.cu
    HEADERS Game.cuh
    LIBS Utils
)