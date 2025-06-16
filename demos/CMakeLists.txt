if(BUILD_DEMO)
    if(DEBUG_MSG)
        message(STATUS "---")
        message(STATUS "BUILD_DEMO is TRUE. Configuring demo executables...")
    endif()

    # ////// BRUTEFORCE //////
    if(DEBUG_MSG)
        message(STATUS "---")
        message(STATUS "## Demo: BruteForce")
    endif()

    if(DEBUG_MSG)
        message(STATUS "Attempting to add executable: demo_bruteforce")
    endif()
    add_executable(demo_bruteforce demo_bruteforce.cpp)
    if(DEBUG_MSG)
        message(STATUS "Executable demo_bruteforce added.")
    endif()

    if(DEBUG_MSG)
        message(STATUS "Linking libraries for demo_bruteforce...")
    endif()
    target_link_libraries(demo_bruteforce PRIVATE dino_lib commons_lib)
    if(DEBUG_MSG)
        message(STATUS "Libraries linked for demo_bruteforce.")
    endif()

    if(DEBUG_MSG)
        message(STATUS "Adding include directories for demo_bruteforce...")
    endif()
    target_include_directories(demo_bruteforce PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../lib
        ${CMAKE_CURRENT_SOURCE_DIR}/../commons
    )
    if(DEBUG_MSG)
        message(STATUS "Include directories added for demo_bruteforce.")
    endif()
    
    # ////// LBBRUTEFORCE //////
    if(DEBUG_MSG)
        message(STATUS "---")
        message(STATUS "## Demo: LbBruteforce")
    endif()

    if(DEBUG_MSG)
        message(STATUS "Attempting to add executable: demo_LbBruteforce")
    endif()
    add_executable(demo_LbBruteforce demo_LbBruteforce.cpp)
    if(DEBUG_MSG)
        message(STATUS "Executable demo_LbBruteforce added.")
    endif()

    if(DEBUG_MSG)
        message(STATUS "Linking libraries for demo_LbBruteforce...")
    endif()
    target_link_libraries(demo_LbBruteforce PRIVATE dino_lib commons_lib)
    if(DEBUG_MSG)
        message(STATUS "Libraries linked for demo_LbBruteforce.")
    endif()

    if(DEBUG_MSG)
        message(STATUS "Adding include directories for demo_LbBruteforce...")
    endif()
    target_include_directories(demo_LbBruteforce PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../lib
        ${CMAKE_CURRENT_SOURCE_DIR}/../commons
    )
    if(DEBUG_MSG)
        message(STATUS "Include directories added for demo_LbBruteforce.")
    endif()

    # ////// MESSI //////
    if(DEBUG_MSG)
        message(STATUS "---")
        message(STATUS "## Demo: Messi")
    endif()

    if(DEBUG_MSG)
        message(STATUS "Attempting to add executable: demo_Messi")
    endif()
    add_executable(demo_Messi demo_Messi.cpp)
    if(DEBUG_MSG)
        message(STATUS "Executable demo_Messi added.")
    endif()

    if(DEBUG_MSG)
        message(STATUS "Linking libraries for demo_Messi...")
    endif()
    target_link_libraries(demo_Messi PRIVATE dino_lib commons_lib)
    if(DEBUG_MSG)
        message(STATUS "Libraries linked for demo_Messi.")
    endif()

    if(DEBUG_MSG)
        message(STATUS "Adding include directories for demo_Messi...")
    endif()
    target_include_directories(demo_Messi PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../lib
        ${CMAKE_CURRENT_SOURCE_DIR}/../commons
    )
    if(DEBUG_MSG)
        message(STATUS "Include directories added for demo_Messi.")
    endif()

    # ////// ODYSSEY //////
    if(DEBUG_MSG)
        message(STATUS "---")
        message(STATUS "## Demo: Odyssey (conditional on MPI)")
    endif()

    if (ODYSSEY_MPI_AVAILABLE)
        if(DEBUG_MSG)
            message(STATUS "ODYSSEY_MPI_AVAILABLE is TRUE. Building demo_Odyssey.")
            message(STATUS "Attempting to add executable: demo_Odyssey")
        endif()
        add_executable(demo_Odyssey demo_Odyssey.cpp)
        if(DEBUG_MSG)
            message(STATUS "Executable demo_Odyssey added.")
        endif()

        if(DEBUG_MSG)
            message(STATUS "Linking libraries for demo_Odyssey...")
        endif()
        target_link_libraries(demo_Odyssey PRIVATE dino_lib commons_lib MPI::MPI_CXX) # Link with MPI for this demo
        if(DEBUG_MSG)
            message(STATUS "Libraries linked for demo_Odyssey.")
        endif()

        if(DEBUG_MSG)
            message(STATUS "Adding include directories for demo_Odyssey...")
        endif()
        target_include_directories(demo_Odyssey PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/../lib
            ${CMAKE_CURRENT_SOURCE_DIR}/../commons
        )
        if(DEBUG_MSG)
            message(STATUS "Include directories added for demo_Odyssey.")
        endif()
    else()
        if(DEBUG_MSG)
            message(STATUS "ODYSSEY_MPI_AVAILABLE is FALSE. demo_Odyssey (MPI-dependent) will NOT be built.")
        endif()
    endif()

    # ////// PARIS //////

    # ////// SING //////

else()
    if(DEBUG_MSG)
        message(STATUS "BUILD_DEMO is FALSE. Demo executables will NOT be built.")
    endif()
endif()