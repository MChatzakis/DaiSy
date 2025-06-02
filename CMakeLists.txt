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

# Add option to build Python bindings 
option(BUILD_PYTHON "Build Python bindings using pybind11" ON)

# Add option to build benchmarks
option(BUILD_BENCHMARK "Build benchmark executables" ON)

# Add option to build Demos files
option(BUILD_DEMO "Build the demo folder" ON)

# Add common code
add_subdirectory(commons)

# Add library
add_subdirectory(lib)

# Add tests
add_subdirectory(tests)

# Conditionally add Python bindings
if(BUILD_PYTHON)
    # Add pybind11
    add_subdirectory(extern/pybind11)

    # Build the Python module
    pybind11_add_module(diNoSimilaritySearch pybinds/setup.cpp)
    target_link_libraries(diNoSimilaritySearch PRIVATE dino_lib)

    # Set output directory for the .so file
    set_target_properties(diNoSimilaritySearch PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/demos"
    )
endif()

# Conditionally add benchmark
if(BUILD_BENCHMARK)
    add_subdirectory(benchmark)
endif()

# Conditionally add demos
if(BUILD_DEMO)
    add_subdirectory(demos)
endif()
