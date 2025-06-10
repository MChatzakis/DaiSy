# lib/CMakeLists.txt

# Create a library target from source files
add_library(
    dino_lib
    distance_computers/DistanceComputer.cpp
    isax/SAX.cpp
    isax/iSAXIndex.cpp
    # Removed SAXBreakpoints.hpp as it's a header, not a source file for compilation
    isax/iSAXPqueue.cpp
    algos/OdysseySearch.cpp # ADDED: Include the source file for OdysseySearch
)

# Make it position-independent for use in a shared object
set_target_properties(dino_lib PROPERTIES POSITION_INDEPENDENT_CODE ON)

# Include all headers (including subfolders)
target_include_directories(
    dino_lib PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/algos # Ensure algos directory headers are found
    ${CMAKE_CURRENT_SOURCE_DIR}/distance_computers
    ${CMAKE_CURRENT_SOURCE_DIR}/isax
)

# ADDED: Conditionally add MPI includes and link MPI if ODYSSEY_MPI_AVAILABLE is true
if (ODYSSEY_MPI_AVAILABLE)
    message(STATUS "Adding MPI includes and libraries to dino_lib.")
    
    # Add MPI include directories so the compiler can find <mpi.h>
    target_include_directories(dino_lib PUBLIC
        ${MPI_C_INCLUDE_DIRS} # Use C_INCLUDE_DIRS (or CXX_INCLUDE_DIRS) for C++ projects
    )
    
    # Link dino_lib with MPI libraries if its components (like Odyssey) use MPI functions
    target_link_libraries(dino_lib PUBLIC
        ${MPI_C_LIBRARIES} # Use C_LIBRARIES (or CXX_LIBRARIES), or MPI::MPI_CXX for modern CMake
    )
endif()

# The add_subdirectory(algos) might be redundant if all algos are explicitly listed above,
# but if it adds other things like tests or separate libraries within 'algos', keep it.
# If 'algos' only contains source files for 'dino_lib', you might consider removing it.
add_subdirectory(algos) 