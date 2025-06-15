cmake_minimum_required(VERSION 3.16)
project(DiNoLibProject)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Compiler flags
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

# Set option
option(BUILD_PYTHON "Build Python bindings using pybind11" ON)
option(BUILD_BENCHMARK "Build benchmark executables" ON)
option(BUILD_DEMO "Build the demo folder" ON)
option(ODYSSEY_MPI "Build Odyssey components with MPI support" ON)

# Initialize ODYSSEY_MPI_AVAILABLE to FALSE by default.
# It's kept as CACHE INTERNAL for user persistency, but its value will be updated.
set(ODYSSEY_MPI_AVAILABLE FALSE CACHE INTERNAL "Internal flag indicating if MPI is available.")

if (ODYSSEY_MPI)
    message(STATUS "Attempting to build with MPI support (ODYSSEY_MPI is ON by default)...")
    find_package(MPI QUIET) # Changed to QUIET, so it doesn't fail immediately if not found

    if (MPI_FOUND)
        message(STATUS "MPI found and will be used:")
        message(STATUS "  C Compiler: ${MPI_C_COMPILER}")
        message(STATUS "  C++ Compiler: ${MPI_CXX_COMPILER}")
        message(STATUS "  Includes: ${MPI_C_INCLUDE_DIRS}")
        message(STATUS "  Libraries: ${MPI_C_LIBRARIES}")

        # Set the flag to TRUE if MPI was found. This update needs to happen BEFORE subdirectories.
        set(ODYSSEY_MPI_AVAILABLE TRUE CACHE INTERNAL "Internal flag indicating MPI is available.")

    else()
        message(WARNING "MPI not found despite ODYSSEY_MPI being ON. Odyssey MPI components will NOT be built.")
        message(WARNING "To resolve this, please install an MPI implementation (e.g., OpenMPI, MPICH) or set ODYSSEY_MPI=OFF.")
        # If MPI is not found, we explicitly set the flag back to FALSE to prevent issues.
        set(ODYSSEY_MPI_AVAILABLE FALSE CACHE INTERNAL "Internal flag indicating MPI is available.")
    endif()
else()
    message(STATUS "Building without MPI support (ODYSSEY_MPI is OFF).")
    # Ensure this is set if the option is OFF, before subdirectories.
    set(ODYSSEY_MPI_AVAILABLE FALSE CACHE INTERNAL "Internal flag indicating MPI is available.")
endif()


# Now add subdirectories which will correctly see the updated ODYSSEY_MPI_AVAILABLE variable.
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

    if (ODYSSEY_MPI_AVAILABLE)
        message(STATUS "Linking Python module 'diNoSimilaritySearch' with MPI.")
        target_link_libraries(diNoSimilaritySearch PRIVATE ${MPI_CXX_LIBRARIES})
        target_include_directories(diNoSimilaritySearch PRIVATE ${MPI_C_INCLUDE_DIRS})
    endif()    

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