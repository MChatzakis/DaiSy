if(DEBUG_MSG)
    message(STATUS "--- Configuring dino_lib ---")
endif()

# Create a library target from source files
if(DEBUG_MSG)
    message(STATUS "Attempting to add library: dino_lib")
endif()
add_library(
    dino_lib
    distance_computers/DistanceComputer.cpp
    isax/SAX.cpp
    isax/iSAXIndex.cpp
    isax/iSAXPqueue.cpp
    utils/TimerManager.cpp
    # Note: Odyssey.cpp and related files are added via target_sources in algos/CMakeLists.txt
)
if(DEBUG_MSG)
    message(STATUS "Library dino_lib added.")
endif()

# Make it position-independent for use in a shared object
if(DEBUG_MSG)
    message(STATUS "Setting POSITION_INDEPENDENT_CODE property for dino_lib...")
endif()
set_target_properties(dino_lib PROPERTIES POSITION_INDEPENDENT_CODE ON)
if(DEBUG_MSG)
    message(STATUS "POSITION_INDEPENDENT_CODE set for dino_lib.")
endif()

# Include all headers (including subfolders)
if(DEBUG_MSG)
    message(STATUS "Setting public include directories for dino_lib...")
endif()
target_include_directories(
    dino_lib PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/algos # Ensure algos directory headers are found
    ${CMAKE_CURRENT_SOURCE_DIR}/distance_computers
    ${CMAKE_CURRENT_SOURCE_DIR}/isax
    ${CMAKE_SOURCE_DIR}/commons  # for common.hpp (initialize_basis_function)
)
target_link_libraries(dino_lib PUBLIC commons_lib)
if(DEBUG_MSG)
    message(STATUS "Public include directories set for dino_lib.")
endif()

if (BUILD_ODYSSEY_AVAILABLE)
    if(DEBUG_MSG)
        message(STATUS "BUILD_ODYSSEY_AVAILABLE is TRUE. Adding MPI includes and libraries to dino_lib.")
    endif()

    # Add MPI include directories so the compiler can find <mpi.h>
    if(DEBUG_MSG)
        message(STATUS "Adding MPI include directories to dino_lib...")
    endif()
    target_include_directories(dino_lib PUBLIC
        ${MPI_CXX_INCLUDE_DIRS} # Use C_INCLUDE_DIRS (or CXX_INCLUDE_DIRS) for C++ projects
    )
    if(DEBUG_MSG)
        message(STATUS "MPI include directories added to dino_lib.")
    endif()

    # Link dino_lib with MPI libraries if its components (like Odyssey) use MPI functions
    if(DEBUG_MSG)
        message(STATUS "Linking dino_lib with MPI libraries...")
    endif()
    target_link_libraries(dino_lib PUBLIC
        ${MPI_CXX_LIBRARIES} 
    )
    if(DEBUG_MSG)
        message(STATUS "MPI libraries linked to dino_lib.")
    endif()
else()
    if(DEBUG_MSG)
        message(STATUS "BUILD_ODYSSEY_AVAILABLE is FALSE. MPI dependencies will NOT be added to dino_lib.")
    endif()
endif()

if(DEBUG_MSG)
    message(STATUS "Adding subdirectory: algos")
endif()
add_subdirectory(algos)
if(DEBUG_MSG)
    message(STATUS "Subdirectory 'algos' added.")
endif()

if(DEBUG_MSG)
    message(STATUS "--- dino_lib Configuration Complete ---")
endif()