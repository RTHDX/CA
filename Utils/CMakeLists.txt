cmake_minimum_required(VERSION 3.15)
project(Utils LANGUAGES CXX CUDA)

create_library(
    TARGET ${PROJECT_NAME}
    SOURCES Utils.cpp
    HEADERS Utils.hpp
)
