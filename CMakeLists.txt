cmake_minimum_required(VERSION 3.16)
project(solver)

set(CMAKE_CXX_STANDARD 14)

find_package(OpenCL REQUIRED)

add_executable(sparse_solver main.cpp mmio.c)
target_link_libraries (sparse_solver ${OpenCL_LIBRARY})