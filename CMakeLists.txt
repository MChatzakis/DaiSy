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
option(BUILD_PYTHON "Build Python bindings using pybind11" OFF)
option(BUILD_BENCHMARK "Build benchmark executables" OFF)
option(BUILD_DEMO "Build the demo folder" ON)
option(BUILD_TESTS "Build test executables" OFF)
option(BUILD_ODYSSEY "Build Odyssey components with MPI support" OFF)
option(ODYSSEY_DEBUG_RESULTS "Print Odyssey search results (query_result) for debug before copy to I,D" OFF)
option(BUILD_SING "Enable CUDA support" OFF)

# Backward compatibility: ENABLE_MPI and ODYSSEY_MPI map to BUILD_ODYSSEY
if(DEFINED ENABLE_MPI)
    if(ENABLE_MPI STREQUAL "OFF" OR NOT ENABLE_MPI)
        set(BUILD_ODYSSEY OFF CACHE BOOL "Build Odyssey components with MPI support" FORCE)
        if(DEBUG_MSG)
            message(STATUS "ENABLE_MPI=OFF detected, setting BUILD_ODYSSEY=OFF")
        endif()
    endif()
endif()

if(DEFINED ODYSSEY_MPI)
    if(ODYSSEY_MPI STREQUAL "OFF" OR NOT ODYSSEY_MPI)
        set(BUILD_ODYSSEY OFF CACHE BOOL "Build Odyssey components with MPI support" FORCE)
        if(DEBUG_MSG)
            message(STATUS "ODYSSEY_MPI=OFF detected (deprecated), setting BUILD_ODYSSEY=OFF")
        endif()
    elseif(ODYSSEY_MPI STREQUAL "ON" OR ODYSSEY_MPI)
        set(BUILD_ODYSSEY ON CACHE BOOL "Build Odyssey components with MPI support" FORCE)
        if(DEBUG_MSG)
            message(STATUS "ODYSSEY_MPI=ON detected (deprecated), setting BUILD_ODYSSEY=ON")
        endif()
    endif()
endif()

message(STATUS "  DEBUG_MSG: ${DEBUG_MSG}")
message(STATUS "Build options:")
message(STATUS "  BUILD_PYTHON: ${BUILD_PYTHON}")
message(STATUS "  BUILD_BENCHMARK: ${BUILD_BENCHMARK}")
message(STATUS "  BUILD_DEMO: ${BUILD_DEMO}")
message(STATUS "  BUILD_TESTS: ${BUILD_TESTS}")
message(STATUS "  BUILD_ODYSSEY: ${BUILD_ODYSSEY}")
message(STATUS "  ODYSSEY_DEBUG_RESULTS: ${ODYSSEY_DEBUG_RESULTS}")
message(STATUS "  BUILD_SING: ${BUILD_SING}")

# Use ODYSSEY_DEBUG_RESULTS so CMake does not warn "variable not used", and propagate to C++
if(ODYSSEY_DEBUG_RESULTS)
    add_compile_definitions(ODYSSEY_DEBUG_RESULTS=1)
endif()

# Initialize BUILD_ODYSSEY_AVAILABLE to FALSE by default.
# It's kept as CACHE INTERNAL for user persistency, but its value will be updated.
set(BUILD_ODYSSEY_AVAILABLE FALSE CACHE INTERNAL "Internal flag indicating if MPI is available.")

if(DEBUG_MSG)
    message(DEBUG "BUILD_ODYSSEY_AVAILABLE initialized to: ${BUILD_ODYSSEY_AVAILABLE}")
endif()

if (BUILD_ODYSSEY)
    if(DEBUG_MSG)
        message(STATUS "Attempting to build with MPI support (BUILD_ODYSSEY is ON)...")
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
        set(BUILD_ODYSSEY_AVAILABLE TRUE CACHE INTERNAL "Internal flag indicating MPI is available.")
        if(DEBUG_MSG)
            message(DEBUG "BUILD_ODYSSEY_AVAILABLE updated to: ${BUILD_ODYSSEY_AVAILABLE} (MPI_FOUND)")
        endif()

    else()
        if(DEBUG_MSG)
            message(WARNING "MPI not found despite BUILD_ODYSSEY being ON. Odyssey MPI components will NOT be built.")
            message(WARNING "To resolve this, please install an MPI implementation (e.g., OpenMPI, MPICH) or set BUILD_ODYSSEY=OFF.")
        endif()
        # If MPI is not found, we explicitly set the flag back to FALSE to prevent issues.
        set(BUILD_ODYSSEY_AVAILABLE FALSE CACHE INTERNAL "Internal flag indicating MPI is available.")

        if(DEBUG_MSG)
            message(DEBUG "BUILD_ODYSSEY_AVAILABLE updated to: ${BUILD_ODYSSEY_AVAILABLE} (MPI_NOT_FOUND)")
        endif()
    endif()
else()
    if(DEBUG_MSG)
        message(STATUS "Building without MPI support (BUILD_ODYSSEY is OFF).")
    endif()
    # Ensure this is set if the option is OFF, before subdirectories.
    set(BUILD_ODYSSEY_AVAILABLE FALSE CACHE INTERNAL "Internal flag indicating MPI is available.")
    
    if(DEBUG_MSG)
        message(DEBUG "BUILD_ODYSSEY_AVAILABLE confirmed as: ${BUILD_ODYSSEY_AVAILABLE} (BUILD_ODYSSEY OFF)")
    endif()
endif()

# Initialize BUILD_SING_AVAILABLE to FALSE by default
set(BUILD_SING_AVAILABLE FALSE CACHE INTERNAL "Internal flag indicating if CUDA is available.")

if(DEBUG_MSG)
    message(DEBUG "BUILD_SING_AVAILABLE initialized to: ${BUILD_SING_AVAILABLE}")
endif()

# Support both ENABLE_CUDA and SING_CUDA for backward compatibility, mapping to BUILD_SING
if(DEFINED ENABLE_CUDA)
    if(ENABLE_CUDA STREQUAL "OFF" OR NOT ENABLE_CUDA)
        set(BUILD_SING OFF CACHE BOOL "Enable CUDA support" FORCE)
        if(DEBUG_MSG)
            message(STATUS "ENABLE_CUDA=OFF detected, setting BUILD_SING=OFF")
        endif()
    endif()
endif()

if(DEFINED SING_CUDA)
    if(SING_CUDA STREQUAL "OFF" OR NOT SING_CUDA)
        set(BUILD_SING OFF CACHE BOOL "Enable CUDA support" FORCE)
        if(DEBUG_MSG)
            message(STATUS "SING_CUDA=OFF detected (deprecated), setting BUILD_SING=OFF")
        endif()
    elseif(SING_CUDA STREQUAL "ON" OR SING_CUDA)
        set(BUILD_SING ON CACHE BOOL "Enable CUDA support" FORCE)
        if(DEBUG_MSG)
            message(STATUS "SING_CUDA=ON detected (deprecated), setting BUILD_SING=ON")
        endif()
    endif()
endif()

if (BUILD_SING)
    if(DEBUG_MSG)
        message(STATUS "Attempting to build with CUDA support (BUILD_SING is ON)...")
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
        set(BUILD_SING_AVAILABLE TRUE CACHE INTERNAL "Internal flag indicating CUDA is available." FORCE)
        if(DEBUG_MSG)
            message(STATUS "CUDA found and will be used for SING components.")
            message(DEBUG "CUDAToolkit_INCLUDE_DIRS: ${CUDAToolkit_INCLUDE_DIRS}")
            message(DEBUG "CUDAToolkit_LIBRARIES: ${CUDAToolkit_LIBRARIES}")
            message(DEBUG "BUILD_SING_AVAILABLE updated to: ${BUILD_SING_AVAILABLE} (CUDA_FOUND)")
        endif()
    else()
        set(BUILD_SING_AVAILABLE FALSE CACHE INTERNAL "Internal flag indicating CUDA is available." FORCE)
        if(DEBUG_MSG)
            message(WARNING "CUDA not found despite BUILD_SING being ON. SING CUDA components will NOT be built.")
            message(DEBUG "BUILD_SING_AVAILABLE updated to: ${BUILD_SING_AVAILABLE} (CUDA_NOT_FOUND)")
        endif()
    endif()
else()
    set(BUILD_SING_AVAILABLE FALSE CACHE INTERNAL "Internal flag indicating CUDA is available." FORCE)
    if(DEBUG_MSG)
        message(STATUS "Building without CUDA support (BUILD_SING is OFF).")
        message(DEBUG "BUILD_SING_AVAILABLE confirmed as: ${BUILD_SING_AVAILABLE} (BUILD_SING OFF)")
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
    pybind11_add_module(daisy pybinds/setup.cpp)
    if(DEBUG_MSG)
        message(STATUS "Python module 'daisy' created from pybinds/setup.cpp.")
    endif()
    
    target_link_libraries(daisy PRIVATE dino_lib)
    if(DEBUG_MSG)
        message(STATUS "Python module 'daisy' linked to 'dino_lib'.")
    endif()

    # Pass BUILD_SING_ENABLED definition to Python module
    if (BUILD_SING_AVAILABLE)
        target_compile_definitions(daisy PRIVATE SING_CUDA_ENABLED=1)
        if(DEBUG_MSG)
            message(STATUS "Python module 'daisy' compiled with SING_CUDA_ENABLED=1.")
        endif()
    else()
        target_compile_definitions(daisy PRIVATE SING_CUDA_ENABLED=0)
        if(DEBUG_MSG)
            message(STATUS "Python module 'daisy' compiled with SING_CUDA_ENABLED=0.")
        endif()
    endif()

    if (BUILD_ODYSSEY_AVAILABLE)
        if(DEBUG_MSG)
            message(STATUS "Linking Python module 'daisy' with MPI.")
        endif()
        target_link_libraries(daisy PRIVATE ${MPI_CXX_LIBRARIES})
        target_include_directories(daisy PRIVATE ${MPI_CXX_INCLUDE_DIRS})
        if(DEBUG_MSG)
            message(DEBUG "MPI Libraries: ${MPI_CXX_LIBRARIES}")
            message(DEBUG "MPI Includes: ${MPI_CXX_INCLUDE_DIRS}")
        endif()
    else()
        if(DEBUG_MSG)
            message(STATUS "BUILD_ODYSSEY_AVAILABLE is FALSE. Python module 'daisy' will NOT be linked with MPI.")
        endif()
    endif()    

    # Set output directory for the .so file (demos/ so that import daisy finds daisy.*.so when run from demos)
    set_target_properties(daisy PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/demos"
    )
    if(DEBUG_MSG)
        message(STATUS "Python module 'daisy' output directory set to: ${CMAKE_SOURCE_DIR}/demos")
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
