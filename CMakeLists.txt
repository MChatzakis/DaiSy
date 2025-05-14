cmake_minimum_required(VERSION 3.16)
project(DiNoLibProject)

set(CMAKE_CXX_STANDARD 17)

add_compile_options(-march=native -mavx -mavx2 -msse3 -O2 -g -fopenmp -Wuninitialized -Wunused-variable -Wparentheses)

# Add common code
add_subdirectory(commons)

# Add library
add_subdirectory(lib)

# Add demos
add_subdirectory(demos)

# Add tests
add_subdirectory(tests)

