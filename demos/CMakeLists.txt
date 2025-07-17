if(BUILD_DEMO)
    if(DEBUG_MSG)
        message(STATUS "---")
        message(STATUS "BUILD_DEMO is TRUE. Configuring demo executables...")
    endif()

    # ////// BRUTEFORCE L2Square //////
    if(DEBUG_MSG)
        message(STATUS "---")
        message(STATUS "## Demo: BruteForce")
    endif()

    if(DEBUG_MSG)
        message(STATUS "Attempting to add executable: demo_bruteforce_L2Square")
    endif()
    add_executable(demo_bruteforce_L2Square demo_bruteforce_L2Square.cpp)
    if(DEBUG_MSG)
        message(STATUS "Executable demo_bruteforce_L2Square added.")
    endif()

    if(DEBUG_MSG)
        message(STATUS "Linking libraries for demo_bruteforce_L2Square...")
    endif()
    target_link_libraries(demo_bruteforce_L2Square PRIVATE dino_lib commons_lib)
    if(DEBUG_MSG)
        message(STATUS "Libraries linked for demo_bruteforce_L2Square.")
    endif()

    if(DEBUG_MSG)
        message(STATUS "Adding include directories for demo_bruteforce_L2Square...")
    endif()
    target_include_directories(demo_bruteforce_L2Square PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../lib
        ${CMAKE_CURRENT_SOURCE_DIR}/../commons
    )
    if(DEBUG_MSG)
        message(STATUS "Include directories added for demo_bruteforce_L2Square.")
    endif()

    # ////// BRUTEFORCE DTW //////
    if(DEBUG_MSG)
        message(STATUS "---")
        message(STATUS "## Demo: BruteForce DTW")
    endif()

    if(DEBUG_MSG)
        message(STATUS "Attempting to add executable: demo_bruteforce_dtw")
    endif()
    add_executable(demo_bruteforce_dtw demo_bruteforce_dtw.cpp)
    if(DEBUG_MSG)
        message(STATUS "Executable demo_bruteforce_dtw added.")
    endif()

    if(DEBUG_MSG)
        message(STATUS "Linking libraries for demo_bruteforce_dtw...")
    endif()
    target_link_libraries(demo_bruteforce_dtw PRIVATE dino_lib commons_lib)
    if(DEBUG_MSG)
        message(STATUS "Libraries linked for demo_bruteforce_dtw.")
    endif()

    if(DEBUG_MSG)
        message(STATUS "Adding include directories for demo_bruteforce_dtw...")
    endif()
    target_include_directories(demo_bruteforce_dtw PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../lib
        ${CMAKE_CURRENT_SOURCE_DIR}/../commons
    )
    if(DEBUG_MSG)
        message(STATUS "Include directories added for demo_bruteforce_dtw.")
    endif()
    
    # ////// LBBRUTEFORCE L2Square //////
    if(DEBUG_MSG)
        message(STATUS "---")
        message(STATUS "## Demo: LbBruteforce")
    endif()

    if(DEBUG_MSG)
        message(STATUS "Attempting to add executable: demo_LbBruteforce_L2Square")
    endif()
    add_executable(demo_LbBruteforce_L2Square demo_LbBruteforce_L2Square.cpp)
    if(DEBUG_MSG)
        message(STATUS "Executable demo_LbBruteforce_L2Square added.")
    endif()

    if(DEBUG_MSG)
        message(STATUS "Linking libraries for demo_LbBruteforce_L2Square...")
    endif()
    target_link_libraries(demo_LbBruteforce_L2Square PRIVATE dino_lib commons_lib)
    if(DEBUG_MSG)
        message(STATUS "Libraries linked for demo_LbBruteforce_L2Square.")
    endif()

    if(DEBUG_MSG)
        message(STATUS "Adding include directories for demo_LbBruteforce_L2Square...")
    endif()
    target_include_directories(demo_LbBruteforce_L2Square PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../lib
        ${CMAKE_CURRENT_SOURCE_DIR}/../commons
    )
    if(DEBUG_MSG)
        message(STATUS "Include directories added for demo_LbBruteforce_L2Square.")
    endif()

    # ////// MESSI L2Square //////
    if(DEBUG_MSG)
        message(STATUS "---")
        message(STATUS "## Demo: Messi")
    endif()

    if(DEBUG_MSG)
        message(STATUS "Attempting to add executable: demo_Messi_L2Square")
    endif()
    add_executable(demo_Messi_L2Square demo_Messi_L2Square.cpp)
    if(DEBUG_MSG)
        message(STATUS "Executable demo_Messi_L2Square added.")
    endif()

    if(DEBUG_MSG)
        message(STATUS "Linking libraries for demo_Messi_L2Square...")
    endif()
    target_link_libraries(demo_Messi_L2Square PRIVATE dino_lib commons_lib)
    if(DEBUG_MSG)
        message(STATUS "Libraries linked for demo_Messi_L2Square.")
    endif()

    if(DEBUG_MSG)
        message(STATUS "Adding include directories for demo_Messi_L2Square...")
    endif()
    target_include_directories(demo_Messi_L2Square PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../lib
        ${CMAKE_CURRENT_SOURCE_DIR}/../commons
    )
    if(DEBUG_MSG)
        message(STATUS "Include directories added for demo_Messi_L2Square.")
    endif()

# ////// ODYSSEY L2Square //////
    if(DEBUG_MSG)
        message(STATUS "---")
        message(STATUS "## Demo: Odyssey (conditional on MPI)")
    endif()

    if (ODYSSEY_MPI_AVAILABLE)
        if(DEBUG_MSG)
            message(STATUS "ODYSSEY_MPI_AVAILABLE is TRUE. Building demo_Odyssey_L2Square.")
            message(STATUS "Attempting to add executable: demo_Odyssey_L2Square")
        endif()
        add_executable(demo_Odyssey_L2Square demo_Odyssey_L2Square.cpp)
        if(DEBUG_MSG)
            message(STATUS "Executable demo_Odyssey_L2Square added.")
        endif()

        if(DEBUG_MSG)
            message(STATUS "Linking libraries for demo_Odyssey_L2Square...")
        endif()
        target_link_libraries(demo_Odyssey_L2Square PRIVATE dino_lib commons_lib MPI::MPI_CXX) # Link with MPI for this demo
        if(DEBUG_MSG)
            message(STATUS "Libraries linked for demo_Odyssey_L2Square.")
        endif()

        if(DEBUG_MSG)
            message(STATUS "Adding include directories for demo_Odyssey_L2Square...")
        endif()
        target_include_directories(demo_Odyssey_L2Square PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/../lib
            ${CMAKE_CURRENT_SOURCE_DIR}/../commons
        )
        if(DEBUG_MSG)
            message(STATUS "Include directories added for demo_Odyssey_L2Square.")
        endif()
    else()
        if(DEBUG_MSG)
            message(STATUS "ODYSSEY_MPI_AVAILABLE is FALSE. demo_Odyssey_L2Square (MPI-dependent) will NOT be built.")
        endif()
    endif()

    # ////// PARIS L2Square //////

    # ////// SING L2Square //////
    if(DEBUG_MSG)
        message(STATUS "---")
        message(STATUS "## Demo: Sing")
    endif()

    if(SING_CUDA_AVAILABLE)
        if(DEBUG_MSG)
            message(STATUS "SING_CUDA_AVAILABLE is TRUE. Building demo_Sing_L2Square.")
        endif()

        add_executable(demo_Sing_L2Square demo_Sing_L2Square.cpp)

        target_link_libraries(demo_Sing_L2Square PRIVATE dino_lib commons_lib CUDA::cudart)

        target_include_directories(demo_Sing_L2Square PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/../lib
            ${CMAKE_CURRENT_SOURCE_DIR}/../commons
        )
    else()
        if(DEBUG_MSG)
            message(STATUS "SING_CUDA_AVAILABLE is FALSE. demo_Sing_L2Square will NOT be built.")
        endif()
    endif()

else()
    if(DEBUG_MSG)
        message(STATUS "BUILD_DEMO is FALSE. Demo executables will NOT be built.")
    endif()
endif()