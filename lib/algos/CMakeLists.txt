find_package(OpenMP REQUIRED) 

target_sources(dino_lib 
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/BruteforceSearch.cpp
)

target_sources(dino_lib 
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/LbBruteforceSearch.cpp
)

# target_sources(dino_lib 
#     PRIVATE
#     ${CMAKE_CURRENT_SOURCE_DIR}/MessiSearch.cpp
# )

# ONLY add MPI includes and link MPI if ODYSSEY_MPI_AVAILABLE is true
if (ODYSSEY_MPI_AVAILABLE)
    message(STATUS "Adding MPI includes and libraries to dino_lib.")
    
    # Add MPI include directories so the compiler can find <mpi.h>
    target_include_directories(dino_lib PUBLIC
        ${MPI_C_INCLUDE_DIRS} # Use C_INCLUDE_DIRS or CXX_INCLUDE_DIRS for C++ projects
    )
    
    # Link dino_lib with MPI libraries if its components (like Odyssey) use MPI functions
    target_link_libraries(dino_lib PUBLIC
        ${MPI_C_LIBRARIES} # Use C_LIBRARIES or CXX_LIBRARIES, or MPI::MPI_CXX for modern CMake
    )
endif()

# target_sources(dino_lib 
#     PRIVATE
#     ${CMAKE_CURRENT_SOURCE_DIR}/ParISSearch.cpp
# )

# target_sources(dino_lib 
#     PRIVATE
#     ${CMAKE_CURRENT_SOURCE_DIR}/SingSearch.cpp
# )

if(OpenMP_CXX_FOUND)
    target_compile_options(
        dino_lib 
        PRIVATE 
        ${OpenMP_CXX_FLAGS}
    )

    target_link_libraries(
        dino_lib 
        PRIVATE 
        OpenMP::OpenMP_CXX
    )
    
else()
    message(WARNING "OpenMP not found. Proceeding without OpenMP support.")
endif()