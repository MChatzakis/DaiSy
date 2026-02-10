if(DEBUG_MSG)
    message(STATUS "--- Configuring Algorithms for dino_lib ---")
endif()

if(DEBUG_MSG)
    message(STATUS "Attempting to find OpenMP package...")
endif()
find_package(OpenMP REQUIRED)
if(DEBUG_MSG)
    message(STATUS "OpenMP package search completed.")
endif()

if(DEBUG_MSG)
    message(STATUS "---")
    message(STATUS "## Adding Algorithm Sources to dino_lib")
endif()

# ////// BRUTEFORCE //////
if(DEBUG_MSG)
    message(STATUS "Adding Bruteforce.cpp to dino_lib sources.")
endif()
target_sources(dino_lib
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/Bruteforce.cpp
)
if(DEBUG_MSG)
    message(STATUS "Bruteforce.cpp added.")
endif()

# ////// LBBRUTEFORCE //////
if(DEBUG_MSG)
    message(STATUS "Adding LbBruteforce.cpp to dino_lib sources.")
endif()
target_sources(dino_lib
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/LbBruteforce.cpp
)
if(DEBUG_MSG)
    message(STATUS "LbBruteforce.cpp added.")
endif()

# ////// MESSI //////
if(DEBUG_MSG)
    message(STATUS "Adding Messi.cpp to dino_lib sources (commented out).")
endif()
target_sources(dino_lib
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/Messi.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/../isax/iSAXSearch.cpp
)
if(DEBUG_MSG)
    message(STATUS "Messi.cpp added (commented out).")
endif()

# ////// ODYSSEY //////
if (BUILD_ODYSSEY_AVAILABLE)
    if(DEBUG_MSG)
        message(STATUS "BUILD_ODYSSEY_AVAILABLE is TRUE. Configuring MPI for Odyssey.")
        message(STATUS "Adding MPI C include directories to dino_lib...")
    endif()
    target_include_directories(dino_lib PUBLIC
        ${MPI_C_INCLUDE_DIRS}
    )
    if(DEBUG_MSG)
        message(STATUS "MPI C include directories added.")
    endif()

    if(DEBUG_MSG)
        message(STATUS "Linking MPI C libraries to dino_lib...")
    endif()
    target_link_libraries(dino_lib PUBLIC
        ${MPI_C_LIBRARIES}
    )
    if(DEBUG_MSG)
        message(STATUS "MPI C libraries linked.")
    endif()

    target_compile_definitions(dino_lib PUBLIC ODYSSEY_MPI) 
    if(DEBUG_MSG)
        message(STATUS "Defined ODYSSEY_MPI for C++ preprocessor.")
    endif()
else()
    if(DEBUG_MSG)
        message(STATUS "BUILD_ODYSSEY_AVAILABLE is FALSE. MPI configuration for Odyssey skipped.")
    endif()
    target_compile_definitions(dino_lib PUBLIC ODYSSEY_MPI=0)
endif()

if(ODYSSEY_DEBUG_RESULTS)
    target_compile_definitions(dino_lib PRIVATE ODYSSEY_DEBUG_RESULTS=1)
    if(DEBUG_MSG)
        message(STATUS "ODYSSEY_DEBUG_RESULTS enabled: Odyssey will print raw results before copy to I,D.")
    endif()
endif()

# Odyssey sources require MPI; only add them when BUILD_ODYSSEY_AVAILABLE
if(BUILD_ODYSSEY_AVAILABLE)
if(DEBUG_MSG)
    message(STATUS "Adding Odyssey.cpp to dino_lib sources.")
endif()
    target_sources(dino_lib
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/odyssey/Odyssey.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/odyssey/bsf_sharing.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/odyssey/replication.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/odyssey/workstealing.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/odyssey/indexing.cpp
)
if(DEBUG_MSG)
    message(STATUS "Odyssey.cpp, bsf_sharing.cpp, replication.cpp, workstealing.cpp, and indexing.cpp added.")
endif()
else()
if(DEBUG_MSG)
    message(STATUS "ODYSSEY_MPI_AVAILABLE is FALSE. Skipping Odyssey sources (require MPI).")
endif()
endif()

# ////// PARIS //////
if(DEBUG_MSG)
    message(STATUS "Adding ParIS.cpp to dino_lib sources.")
endif()
target_sources(dino_lib
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/ParIS.cpp
)
if(DEBUG_MSG)
    message(STATUS "ParIS.cpp added.")
endif()

# ////// SING //////
if (SING_CUDA_AVAILABLE)
    if(DEBUG_MSG)
        message(STATUS "SING_CUDA_AVAILABLE is TRUE. Configuring CUDA for SING components.")
        message(STATUS "Adding CUDA Toolkit include directories to dino_lib...")
    endif()

    target_include_directories(dino_lib PUBLIC
        ${CUDAToolkit_INCLUDE_DIRS}
    )

    if(DEBUG_MSG)
        message(STATUS "CUDA include directories added.")
    endif()

    if(DEBUG_MSG)
        message(STATUS "Adding Sing.cpp and singlib.cu to dino_lib sources.")
    endif()
    target_sources(dino_lib
        PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/Sing.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/singlib.cu
    )
    set_source_files_properties(${CMAKE_CURRENT_SOURCE_DIR}/singlib.cu PROPERTIES LANGUAGE CUDA)
    if(DEBUG_MSG)
        message(STATUS "Sing.cpp and singlib.cu added to dino_lib sources.")
    endif()

    if(DEBUG_MSG)
        message(STATUS "Linking CUDA Toolkit libraries to dino_lib...")
    endif()

    target_link_libraries(dino_lib PRIVATE CUDA::cudart)

    if(DEBUG_MSG)
        message(STATUS "CUDA libraries linked.")
    endif()
    
    target_compile_definitions(dino_lib PUBLIC SING_CUDA_ENABLED)
    if(DEBUG_MSG)
        message(STATUS "Defined SING_CUDA_ENABLED for C++ preprocessor.")
    endif()
else()
    if(DEBUG_MSG)
        message(STATUS "SING_CUDA_AVAILABLE is FALSE. CUDA configuration for SING skipped.")
    endif()
    target_compile_definitions(dino_lib PUBLIC SING_CUDA_ENABLED=0)
endif()

# /// OpenMP ///
if(OpenMP_CXX_FOUND)
    if(DEBUG_MSG)
        message(STATUS "OpenMP CXX found. Applying compile options to dino_lib.")
    endif()
    target_compile_options(
        dino_lib
        PRIVATE
        ${OpenMP_CXX_FLAGS}
    )
    if(DEBUG_MSG)
        message(STATUS "OpenMP CXX compile options applied.")
    endif()

    if(DEBUG_MSG)
        message(STATUS "Linking OpenMP CXX libraries to dino_lib.")
    endif()
    target_link_libraries(
        dino_lib
        PRIVATE
        OpenMP::OpenMP_CXX
    )
    if(DEBUG_MSG)
        message(STATUS "OpenMP CXX libraries linked.")
    endif()

else()
    if(DEBUG_MSG)
        message(WARNING "OpenMP not found. Proceeding without OpenMP support for dino_lib.")
    endif()
endif()

if(DEBUG_MSG)
    message(STATUS "--- Algorithm Configuration for dino_lib Complete ---")
endif()