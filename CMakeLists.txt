cmake_minimum_required(VERSION 3.16)
project(DiNoLibProject)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_compile_options(-march=native 
    -mavx 
    -mavx2 
    -msse3 
    -O2 
    -g 
    -fopenmp 
    -Wuninitialized 
#    -Wunused-variable 
    -Wparentheses
)

# Add common code
add_subdirectory(commons)

# Add library
add_subdirectory(lib)

# Add demos
add_subdirectory(demos)

# Add tests
add_subdirectory(tests)

# Add pybind11 as submodule
add_subdirectory(extern/pybind11)

# Build the Python module
pybind11_add_module(diNo_lib pybinds/setup.cpp)
target_link_libraries(diNo_lib PRIVATE dino_lib)

# Set output directory for the .so file
set_target_properties(diNo_lib PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/scripts"
)