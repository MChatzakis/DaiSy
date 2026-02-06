cmake_minimum_required(VERSION 3.16)
project(DiNoLibProject)

message(STATUS "----------------------------------------------------")
message(STATUS "Configuring project: ${CMAKE_PROJECT_NAME}")
message(STATUS "CMake version: ${CMAKE_VERSION}")
message(STATUS "Source directory: ${CMAKE_SOURCE_DIR}")
message(STATUS "Binary directory: ${CMAKE_BINARY_DIR}")
message(STATUS "----------------------------------------------------")

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
message(STATUS "C++ standard set to: ${CMAKE_CXX_STANDARD}")

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
message(STATUS "Common compiler options added.")
message(DEBUG "Compiler options: -march=native -mavx -mavx2 -msse3 -O2 -g -fopenmp -Wuninitialized -Wparentheses")

# Set option
option(DEBUG_MSG "Enable verbose debug messages" OFF)
option(BUILD_PYTHON "Build Python bindings using pybind11" ON)
option(BUILD_BENCHMARK "Build benchmark executables" ON)
option(BUILD_DEMO "Build the demo folder" ON)
option(BUILD_TESTS "Build test executables" ON)
option(ODYSSEY_MPI "Build Odyssey components with MPI support" ON)
option(ODYSSEY_DEBUG_RESULTS "Print Odyssey search results (query_result) for debug before copy to I,D" OFF)
option(SING_CUDA "Enable CUDA support" OFF)

message(STATUS "  DEBUG_MSG: ${DEBUG_MSG}")
message(STATUS "Build options:")
message(STATUS "  BUILD_PYTHON: ${BUILD_PYTHON}")
message(STATUS "  BUILD_BENCHMARK: ${BUILD_BENCHMARK}")
message(STATUS "  BUILD_DEMO: ${BUILD_DEMO}")
message(STATUS "  BUILD_TESTS: ${BUILD_TESTS}")
message(STATUS "  ODYSSEY_MPI: ${ODYSSEY_MPI}")
message(STATUS "  ODYSSEY_DEBUG_RESULTS: ${ODYSSEY_DEBUG_RESULTS}")
message(STATUS "  SING_CUDA: ${SING_CUDA}")

# Use ODYSSEY_DEBUG_RESULTS so CMake does not warn "variable not used", and propagate to C++
if(ODYSSEY_DEBUG_RESULTS)
    add_compile_definitions(ODYSSEY_DEBUG_RESULTS=1)
endif()

# Initialize ODYSSEY_MPI_AVAILABLE to FALSE by default.
# It's kept as CACHE INTERNAL for user persistency, but its value will be updated.
set(ODYSSEY_MPI_AVAILABLE FALSE CACHE INTERNAL "Internal flag indicating if MPI is available.")

if(DEBUG_MSG)
    message(DEBUG "ODYSSEY_MPI_AVAILABLE initialized to: ${ODYSSEY_MPI_AVAILABLE}")
endif()

if (ODYSSEY_MPI)
    if(DEBUG_MSG)
        message(STATUS "Attempting to build with MPI support (ODYSSEY_MPI is ON)...")
    endif()

    find_package(MPI QUIET REQUIRED CXX)

    if (MPI_FOUND)
        if(DEBUG_MSG)
            message(STATUS "MPI found and will be used:")
            message(STATUS "  C++ Compiler: ${MPI_CXX_COMPILER}")
            message(STATUS "  Includes: ${MPI_CXX_INCLUDE_DIRS}")
            message(STATUS "  Libraries: ${MPI_CXX_LIBRARIES}")
        endif()
        
        # Set the flag to TRUE if MPI was found. This update needs to happen BEFORE subdirectories.
        set(ODYSSEY_MPI_AVAILABLE TRUE CACHE INTERNAL "Internal flag indicating MPI is available.")
        if(DEBUG_MSG)
            message(DEBUG "ODYSSEY_MPI_AVAILABLE updated to: ${ODYSSEY_MPI_AVAILABLE} (MPI_FOUND)")
        endif()

    else()
        if(DEBUG_MSG)
            message(WARNING "MPI not found despite ODYSSEY_MPI being ON. Odyssey MPI components will NOT be built.")
            message(WARNING "To resolve this, please install an MPI implementation (e.g., OpenMPI, MPICH) or set ODYSSEY_MPI=OFF.")
        endif()
        # If MPI is not found, we explicitly set the flag back to FALSE to prevent issues.
        set(ODYSSEY_MPI_AVAILABLE FALSE CACHE INTERNAL "Internal flag indicating MPI is available.")

        if(DEBUG_MSG)
            message(DEBUG "ODYSSEY_MPI_AVAILABLE updated to: ${ODYSSEY_MPI_AVAILABLE} (MPI_NOT_FOUND)")
        endif()
    endif()
else()
    if(DEBUG_MSG)
        message(STATUS "Building without MPI support (ODYSSEY_MPI is OFF).")
    endif()
    # Ensure this is set if the option is OFF, before subdirectories.
    set(ODYSSEY_MPI_AVAILABLE FALSE CACHE INTERNAL "Internal flag indicating MPI is available.")
    
    if(DEBUG_MSG)
        message(DEBUG "ODYSSEY_MPI_AVAILABLE confirmed as: ${ODYSSEY_MPI_AVAILABLE} (ODYSSEY_MPI OFF)")
    endif()
endif()

# Initialize SING_CUDA_AVAILABLE to FALSE by default
set(SING_CUDA_AVAILABLE FALSE CACHE INTERNAL "Internal flag indicating if CUDA is available.")

if(DEBUG_MSG)
    message(DEBUG "SING_CUDA_AVAILABLE initialized to: ${SING_CUDA_AVAILABLE}")
endif()

# Support both ENABLE_CUDA and SING_CUDA for backward compatibility
if(DEFINED ENABLE_CUDA)
    if(ENABLE_CUDA STREQUAL "OFF" OR NOT ENABLE_CUDA)
        set(SING_CUDA OFF CACHE BOOL "Enable CUDA support" FORCE)
        if(DEBUG_MSG)
            message(STATUS "ENABLE_CUDA=OFF detected, setting SING_CUDA=OFF")
        endif()
    endif()
endif()

if (SING_CUDA)
    if(DEBUG_MSG)
        message(STATUS "Attempting to build with CUDA support (SING_CUDA is ON)...")
    endif()

    # Check if CUDA is available before enabling the language
    find_package(CUDAToolkit QUIET)
    
    if(CUDAToolkit_FOUND)
        # Specify CUDA architectures to avoid errors
        set(CMAKE_CUDA_ARCHITECTURES 75 CACHE STRING "CUDA architectures to build for")

        if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES OR CMAKE_CUDA_ARCHITECTURES STREQUAL "")
            set(CMAKE_CUDA_ARCHITECTURES 75 CACHE STRING "CUDA architectures to build for" FORCE)
        endif()

        enable_language(CUDA OPTIONAL)
    else()
        if(DEBUG_MSG)
            message(WARNING "CUDA toolkit not found. Skipping CUDA language enable.")
        endif()
    endif()

    if (CUDAToolkit_FOUND)
        set(SING_CUDA_AVAILABLE TRUE CACHE INTERNAL "Internal flag indicating CUDA is available." FORCE)
        if(DEBUG_MSG)
            message(STATUS "CUDA found and will be used for SING components.")
            message(DEBUG "CUDAToolkit_INCLUDE_DIRS: ${CUDAToolkit_INCLUDE_DIRS}")
            message(DEBUG "CUDAToolkit_LIBRARIES: ${CUDAToolkit_LIBRARIES}")
            message(DEBUG "SING_CUDA_AVAILABLE updated to: ${SING_CUDA_AVAILABLE} (CUDA_FOUND)")
        endif()
    else()
        set(SING_CUDA_AVAILABLE FALSE CACHE INTERNAL "Internal flag indicating CUDA is available." FORCE)
        if(DEBUG_MSG)
            message(WARNING "CUDA not found despite SING_CUDA being ON. SING CUDA components will NOT be built.")
            message(DEBUG "SING_CUDA_AVAILABLE updated to: ${SING_CUDA_AVAILABLE} (CUDA_NOT_FOUND)")
        endif()
    endif()
else()
    set(SING_CUDA_AVAILABLE FALSE CACHE INTERNAL "Internal flag indicating CUDA is available." FORCE)
    if(DEBUG_MSG)
        message(STATUS "Building without CUDA support (SING_CUDA is OFF).")
        message(DEBUG "SING_CUDA_AVAILABLE confirmed as: ${SING_CUDA_AVAILABLE} (SING_CUDA OFF)")
    endif()
endif()

# Add subdirectories
if(DEBUG_MSG)
    message(STATUS "----------------------------------------------------")
    message(STATUS "Adding subdirectories...")
endif()

add_subdirectory(commons)
if(DEBUG_MSG)
    message(STATUS "Subdirectory 'commons' added.")
endif()

add_subdirectory(lib)
if(DEBUG_MSG)
    message(STATUS "Subdirectory 'lib' added.")
endif()

if(BUILD_DEMO)
    add_subdirectory(demos)
    if(DEBUG_MSG)
        message(STATUS "Subdirectory 'demos' added.")
    endif()
endif()

if(BUILD_TESTS)
    add_subdirectory(tests)
    if(DEBUG_MSG)
        message(STATUS "Subdirectory 'tests' added.")
    endif()
endif()

# Conditionally add Python bindings
if(BUILD_PYTHON)
    if(DEBUG_MSG)
        message(STATUS "----------------------------------------------------")
        message(STATUS "BUILD_PYTHON is ON. Adding Python bindings.")
    endif()

    # --- ADD THIS SECTION ---
    # Find Python development components *before* adding pybind11
    find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
    if(NOT Python3_FOUND)
        message(FATAL_ERROR "Python 3 development headers not found. Please install python3-dev (or similar) or set BUILD_PYTHON=OFF.")
    endif()
    message(STATUS "Python found: ${Python3_EXECUTABLE}")
    message(STATUS "Python include dirs: ${Python3_INCLUDE_DIRS}")
    message(STATUS "Python libraries: ${Python3_LIBRARIES}")
    # --- END ADDITION ---

    # Add pybind11
    add_subdirectory(extern/pybind11)
    if(DEBUG_MSG)
        message(STATUS "Subdirectory 'extern/pybind11' added.")
    endif()

    # Build the Python module
    pybind11_add_module(diNoSimilaritySearch pybinds/setup.cpp)
    if(DEBUG_MSG)
        message(STATUS "Python module 'diNoSimilaritySearch' created from pybinds/setup.cpp.")
    endif()
    
    target_link_libraries(diNoSimilaritySearch PRIVATE dino_lib)
    if(DEBUG_MSG)
        message(STATUS "Python module 'diNoSimilaritySearch' linked to 'dino_lib'.")
    endif()

    # Pass SING_CUDA_ENABLED definition to Python module
    if (SING_CUDA_AVAILABLE)
        target_compile_definitions(diNoSimilaritySearch PRIVATE SING_CUDA_ENABLED=1)
        if(DEBUG_MSG)
            message(STATUS "Python module 'diNoSimilaritySearch' compiled with SING_CUDA_ENABLED=1.")
        endif()
    else()
        target_compile_definitions(diNoSimilaritySearch PRIVATE SING_CUDA_ENABLED=0)
        if(DEBUG_MSG)
            message(STATUS "Python module 'diNoSimilaritySearch' compiled with SING_CUDA_ENABLED=0.")
        endif()
    endif()

    if (ODYSSEY_MPI_AVAILABLE)
        if(DEBUG_MSG)
            message(STATUS "Linking Python module 'diNoSimilaritySearch' with MPI.")
        endif()
        target_link_libraries(diNoSimilaritySearch PRIVATE ${MPI_CXX_LIBRARIES})
        target_include_directories(diNoSimilaritySearch PRIVATE ${MPI_CXX_INCLUDE_DIRS})
        if(DEBUG_MSG)
            message(DEBUG "MPI Libraries: ${MPI_CXX_LIBRARIES}")
            message(DEBUG "MPI Includes: ${MPI_CXX_INCLUDE_DIRS}")
        endif()
    else()
        if(DEBUG_MSG)
            message(STATUS "ODYSSEY_MPI_AVAILABLE is FALSE. Python module 'diNoSimilaritySearch' will NOT be linked with MPI.")
        endif()
    endif()    

    # Set output directory for the .so file
    set_target_properties(diNoSimilaritySearch PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/demos"
    )
    if(DEBUG_MSG)
        message(STATUS "Python module 'diNoSimilaritySearch' output directory set to: ${CMAKE_SOURCE_DIR}/demos")
    endif()
else()
    if(DEBUG_MSG)
        message(STATUS "BUILD_PYTHON is OFF. Skipping Python bindings.")
    endif()
endif()

# Conditionally add benchmark
if(BUILD_BENCHMARK)
    if(DEBUG_MSG)
        message(STATUS "----------------------------------------------------")
        message(STATUS "BUILD_BENCHMARK is ON. Adding benchmark directory.")
    endif()
    add_subdirectory(benchmark)
    if(DEBUG_MSG)
        message(STATUS "Subdirectory 'benchmark' added.")
    endif()
else()
    if(DEBUG_MSG)
        message(STATUS "BUILD_BENCHMARK is OFF. Skipping benchmark directory.")
    endif()
endif()

# Note: demos subdirectory already added above conditionally

if(DEBUG_MSG)
    message(STATUS "----------------------------------------------------")
    message(STATUS "CMake configuration finished for ${CMAKE_PROJECT_NAME}.")
    message(STATUS "----------------------------------------------------")
endif()
