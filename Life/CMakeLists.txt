cmake_minimum_required(VERSION 3.15)
project(Life LANGUAGES CXX CUDA)

create_executable(
    TARGET ${PROJECT_NAME}
    SOURCES Main.cu
    HEADERS Game.cu
    LIBS Utils
)