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
        message(STATUS "Attempting to add executable: demo_bruteforce_DTW")
    endif()
    add_executable(demo_bruteforce_DTW demo_bruteforce_DTW.cpp)
    if(DEBUG_MSG)
        message(STATUS "Executable demo_bruteforce_DTW added.")
    endif()

    if(DEBUG_MSG)
        message(STATUS "Linking libraries for demo_bruteforce_DTW...")
    endif()
    target_link_libraries(demo_bruteforce_DTW PRIVATE dino_lib commons_lib)
    if(DEBUG_MSG)
        message(STATUS "Libraries linked for demo_bruteforce_DTW.")
    endif()

    if(DEBUG_MSG)
        message(STATUS "Adding include directories for demo_bruteforce_DTW...")
    endif()
    target_include_directories(demo_bruteforce_DTW PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../lib
        ${CMAKE_CURRENT_SOURCE_DIR}/../commons
    )
    if(DEBUG_MSG)
        message(STATUS "Include directories added for demo_bruteforce_DTW.")
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

    # ////// LBBRUTEFORCE DTW //////
    if(DEBUG_MSG)
        message(STATUS "---")
        message(STATUS "## Demo: LbBruteforce DTW")
    endif()

    if(DEBUG_MSG)
        message(STATUS "Attempting to add executable: demo_LbBruteforce_DTW")
    endif()
    add_executable(demo_LbBruteforce_DTW demo_LbBruteforce_DTW.cpp)
    if(DEBUG_MSG)
        message(STATUS "Executable demo_LbBruteforce_DTW added.")
    endif()

    if(DEBUG_MSG)
        message(STATUS "Linking libraries for demo_LbBruteforce_DTW...")
    endif()
    target_link_libraries(demo_LbBruteforce_DTW PRIVATE dino_lib commons_lib)
    if(DEBUG_MSG)
        message(STATUS "Libraries linked for demo_LbBruteforce_DTW.")
    endif()

    if(DEBUG_MSG)
        message(STATUS "Adding include directories for demo_LbBruteforce_DTW...")
    endif()
    target_include_directories(demo_LbBruteforce_DTW PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../lib
        ${CMAKE_CURRENT_SOURCE_DIR}/../commons
    )
    if(DEBUG_MSG)
        message(STATUS "Include directories added for demo_LbBruteforce_DTW.")
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

    # ////// MESSI DTW //////
    if(DEBUG_MSG)
        message(STATUS "---")
        message(STATUS "## Demo: Messi DTW")
    endif()

    if(DEBUG_MSG)
        message(STATUS "Attempting to add executable: demo_Messi_DTW")
    endif()
    add_executable(demo_Messi_DTW demo_Messi_DTW.cpp)
    if(DEBUG_MSG)
        message(STATUS "Executable demo_Messi_DTW added.")
    endif()

    if(DEBUG_MSG)
        message(STATUS "Linking libraries for demo_Messi_DTW...")
    endif()
    target_link_libraries(demo_Messi_DTW PRIVATE dino_lib commons_lib)
    if(DEBUG_MSG)
        message(STATUS "Libraries linked for demo_Messi_DTW.")
    endif()

    if(DEBUG_MSG)
        message(STATUS "Adding include directories for demo_Messi_DTW...")
    endif()
    target_include_directories(demo_Messi_DTW PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../lib
        ${CMAKE_CURRENT_SOURCE_DIR}/../commons
    )
    if(DEBUG_MSG)
        message(STATUS "Include directories added for demo_Messi_DTW.")
    endif()

    # ////// SING DTW //////
    if(BUILD_SING_AVAILABLE)
        if(DEBUG_MSG)
            message(STATUS "---")
            message(STATUS "## Demo: Sing DTW")
            message(STATUS "Attempting to add executable: demo_Sing_DTW")
        endif()
        add_executable(demo_Sing_DTW demo_Sing_DTW.cpp)
        target_link_libraries(demo_Sing_DTW PRIVATE dino_lib commons_lib)
        target_include_directories(demo_Sing_DTW PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/../lib
            ${CMAKE_CURRENT_SOURCE_DIR}/../commons
        )
        if(DEBUG_MSG)
            message(STATUS "Executable demo_Sing_DTW added and configured (CUDA/Sing available).")
        endif()
    else()
        if(DEBUG_MSG)
            message(STATUS "BUILD_SING_AVAILABLE is FALSE. demo_Sing_DTW will NOT be built.")
        endif()
    endif()

# ////// ODYSSEY L2Square //////
    if(DEBUG_MSG)
        message(STATUS "---")
        message(STATUS "## Demo: Odyssey (conditional on MPI)")
    endif()

    if (BUILD_ODYSSEY_AVAILABLE)
        if(DEBUG_MSG)
            message(STATUS "BUILD_ODYSSEY_AVAILABLE is TRUE. Building demo_Odyssey_L2Square.")
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

        # Odyssey DTW (build index + search, same data as ParIS DTW)
        if(DEBUG_MSG)
            message(STATUS "Attempting to add executable: demo_Odyssey_DTW")
        endif()
        add_executable(demo_Odyssey_DTW demo_Odyssey_DTW.cpp)
        if(DEBUG_MSG)
            message(STATUS "Executable demo_Odyssey_DTW added.")
        endif()
        target_link_libraries(demo_Odyssey_DTW PRIVATE dino_lib commons_lib MPI::MPI_CXX)
        target_include_directories(demo_Odyssey_DTW PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/../lib
            ${CMAKE_CURRENT_SOURCE_DIR}/../commons
        )
    else()
        if(DEBUG_MSG)
            message(STATUS "BUILD_ODYSSEY_AVAILABLE is FALSE. demo_Odyssey_L2Square (MPI-dependent) will NOT be built.")
        endif()
    endif()

    # ////// PARIS L2Square //////
    if(DEBUG_MSG)
        message(STATUS "---")
        message(STATUS "## Demo: ParIS")
    endif()

    if(DEBUG_MSG)
        message(STATUS "Attempting to add executable: demo_ParIS_L2Square")
    endif()
    add_executable(demo_ParIS_L2Square demo_ParIS_L2Square.cpp)
    if(DEBUG_MSG)
        message(STATUS "Executable demo_ParIS_L2Square added.")
    endif()

    if(DEBUG_MSG)
        message(STATUS "Linking libraries for demo_ParIS_L2Square...")
    endif()
    target_link_libraries(demo_ParIS_L2Square PRIVATE dino_lib commons_lib)
    if(DEBUG_MSG)
        message(STATUS "Libraries linked for demo_ParIS_L2Square.")
    endif()

    if(DEBUG_MSG)
        message(STATUS "Adding include directories for demo_ParIS_L2Square...")
    endif()
    target_include_directories(demo_ParIS_L2Square PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../lib
        ${CMAKE_CURRENT_SOURCE_DIR}/../commons
    )
    if(DEBUG_MSG)
        message(STATUS "Include directories added for demo_ParIS_L2Square.")
    endif()

    # ////// PARIS DTW //////
    if(DEBUG_MSG)
        message(STATUS "---")
        message(STATUS "## Demo: ParIS DTW")
    endif()

    if(DEBUG_MSG)
        message(STATUS "Attempting to add executable: demo_ParIS_DTW")
    endif()
    add_executable(demo_ParIS_DTW demo_ParIS_DTW.cpp)
    if(DEBUG_MSG)
        message(STATUS "Executable demo_ParIS_DTW added.")
    endif()

    if(DEBUG_MSG)
        message(STATUS "Linking libraries for demo_ParIS_DTW...")
    endif()
    target_link_libraries(demo_ParIS_DTW PRIVATE dino_lib commons_lib)
    if(DEBUG_MSG)
        message(STATUS "Libraries linked for demo_ParIS_DTW.")
    endif()

    if(DEBUG_MSG)
        message(STATUS "Adding include directories for demo_ParIS_DTW...")
    endif()
    target_include_directories(demo_ParIS_DTW PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../lib
        ${CMAKE_CURRENT_SOURCE_DIR}/../commons
    )
    if(DEBUG_MSG)
        message(STATUS "Include directories added for demo_ParIS_DTW.")
    endif()

    # ////// SING L2Square //////
    if(DEBUG_MSG)
        message(STATUS "---")
        message(STATUS "## Demo: Sing L2Square")
    endif()

    if(BUILD_SING_AVAILABLE)
        if(DEBUG_MSG)
            message(STATUS "BUILD_SING_AVAILABLE is TRUE. Building demo_Sing_L2Square.")
        endif()

        add_executable(demo_Sing_L2Square demo_Sing_L2Square.cpp)
        target_link_libraries(demo_Sing_L2Square PRIVATE dino_lib commons_lib)
        target_include_directories(demo_Sing_L2Square PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/../lib
            ${CMAKE_CURRENT_SOURCE_DIR}/../commons
        )
    else()
        if(DEBUG_MSG)
            message(STATUS "BUILD_SING_AVAILABLE is FALSE. demo_Sing_L2Square will NOT be built.")
        endif()
    endif()

else()
    if(DEBUG_MSG)
        message(STATUS "BUILD_DEMO is FALSE. Demo executables will NOT be built.")
    endif()
endif()

