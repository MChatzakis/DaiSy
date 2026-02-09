if(NOT BUILD_BENCHMARK)
    if(DEBUG_MSG)
        message(STATUS "BUILD_BENCHMARK is FALSE. Benchmarks will NOT be built.")
    endif()
    return()
endif()

if(DEBUG_MSG)
    message(STATUS "--- Configuring Benchmarks ---")
    message(STATUS "BUILD_BENCHMARK is TRUE. Proceeding with benchmark configuration.")
endif()

include(FetchContent)
if(DEBUG_MSG)
    message(STATUS "FetchContent module included.")
endif()

FetchContent_Declare(
  benchmark
  GIT_REPOSITORY https://github.com/google/benchmark.git
  GIT_TAG v1.8.3
)
if(DEBUG_MSG)
    message(STATUS "FetchContent_Declare for 'benchmark' completed (Repository: https://github.com/google/benchmark.git, Tag: v1.8.3).")
endif()


FetchContent_MakeAvailable(benchmark)
if(DEBUG_MSG)
    message(STATUS "FetchContent_MakeAvailable for 'benchmark' completed. Google Benchmark is now available.")
endif()

# ////// BRUTEFORCE //////
if(DEBUG_MSG)
    message(STATUS "---")
    message(STATUS "## Benchmark: BruteForce")
endif()

if(DEBUG_MSG)
    message(STATUS "Attempting to add executable: bm_bruteforce_L2Square")
endif()
add_executable(bm_bruteforce_L2Square
    bm_bruteforce_L2Square.cpp
    bm_utils.cpp
    ../commons/paramSetup.cpp
    ../commons/test_bm_utils.cpp
    ../commons/dataloaders.cpp
)
if(DEBUG_MSG)
    message(STATUS "Executable bm_bruteforce_L2Square added.")
endif()

if(DEBUG_MSG)
    message(STATUS "Linking libraries for bm_bruteforce_L2Square...")
endif()
target_link_libraries(bm_bruteforce_L2Square
    PRIVATE
    benchmark::benchmark
    benchmark::benchmark_main
    dino_lib
)
if(DEBUG_MSG)
    message(STATUS "Libraries linked for bm_bruteforce_L2Square.")
endif()

if(DEBUG_MSG)
    message(STATUS "Adding include directories for bm_bruteforce_L2Square...")
endif()
target_include_directories(bm_bruteforce_L2Square
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../lib
    ${CMAKE_CURRENT_SOURCE_DIR}/../commons
)
if(DEBUG_MSG)
    message(STATUS "Include directories added for bm_bruteforce_L2Square.")
endif()

# ////// LBBRUTEFORCE //////
if(DEBUG_MSG)
    message(STATUS "---")
    message(STATUS "## Benchmark: LbBruteforce")
endif()

if(DEBUG_MSG)
    message(STATUS "Attempting to add executable: bm_LbBruteforce_L2Square")
endif()
add_executable(bm_LbBruteforce_L2Square
    bm_LbBruteforce_L2Square.cpp
    bm_utils.cpp
    ../commons/paramSetup.cpp
    ../commons/test_bm_utils.cpp
    ../commons/dataloaders.cpp
)
if(DEBUG_MSG)
    message(STATUS "Executable bm_LbBruteforce_L2Square added.")
endif()

if(DEBUG_MSG)
    message(STATUS "Linking libraries for bm_LbBruteforce_L2Square...")
endif()
target_link_libraries(bm_LbBruteforce_L2Square
    PRIVATE
    benchmark::benchmark
    benchmark::benchmark_main
    dino_lib
)
if(DEBUG_MSG)
    message(STATUS "Libraries linked for bm_LbBruteforce_L2Square.")
endif()

if(DEBUG_MSG)
    message(STATUS "Adding include directories for bm_LbBruteforce_L2Square...")
endif()
target_include_directories(bm_LbBruteforce_L2Square
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../lib
    ${CMAKE_CURRENT_SOURCE_DIR}/../commons
)
if(DEBUG_MSG)
    message(STATUS "Include directories added for bm_LbBruteforce_L2Square.")
endif()

# ////// MESSI //////
if(DEBUG_MSG)
    message(STATUS "---")
    message(STATUS "## Benchmark: Messi")
endif()

if(DEBUG_MSG)
    message(STATUS "Attempting to add executable: bm_Messi_L2Square")
endif()
add_executable(bm_Messi_L2Square
    bm_Messi_L2Square.cpp
    bm_utils.cpp
    ../commons/paramSetup.cpp
    ../commons/test_bm_utils.cpp
    ../commons/dataloaders.cpp
)
if(DEBUG_MSG)
    message(STATUS "Executable bm_Messi_L2Square added.")
endif()

if(DEBUG_MSG)
    message(STATUS "Linking libraries for bm_Messi_L2Square...")
endif()
target_link_libraries(bm_Messi_L2Square
    PRIVATE
    benchmark::benchmark
    benchmark::benchmark_main
    dino_lib
)
if(DEBUG_MSG)
    message(STATUS "Libraries linked for bm_Messi_L2Square.")
endif()

if(DEBUG_MSG)
    message(STATUS "Adding include directories for bm_Messi_L2Square...")
endif()
target_include_directories(bm_Messi_L2Square
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../lib
    ${CMAKE_CURRENT_SOURCE_DIR}/../commons
)
if(DEBUG_MSG)
    message(STATUS "Include directories added for bm_Messi_L2Square.")
endif()

# ////// PARIS //////
if(DEBUG_MSG)
    message(STATUS "---")
    message(STATUS "## Benchmark: ParIS")
endif()

add_executable(bm_ParIS_L2Square
    bm_ParIS_L2Square.cpp
    bm_utils.cpp
    ../commons/paramSetup.cpp
    ../commons/test_bm_utils.cpp
    ../commons/dataloaders.cpp
)
target_link_libraries(bm_ParIS_L2Square
    PRIVATE
    benchmark::benchmark
    benchmark::benchmark_main
    dino_lib
)
target_include_directories(bm_ParIS_L2Square
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../lib
    ${CMAKE_CURRENT_SOURCE_DIR}/../commons
)

# ////// ODYSSEY (conditional on MPI) //////
if(ODYSSEY_MPI_AVAILABLE)
    if(DEBUG_MSG)
        message(STATUS "---")
        message(STATUS "## Benchmark: Odyssey L2Square")
    endif()

    add_executable(bm_Odyssey_L2Square
        bm_Odyssey_L2Square.cpp
        bm_utils.cpp
        ../commons/paramSetup.cpp
        ../commons/test_bm_utils.cpp
        ../commons/dataloaders.cpp
    )
    target_compile_definitions(bm_Odyssey_L2Square PRIVATE ODYSSEY_MPI=1)
    target_link_libraries(bm_Odyssey_L2Square
        PRIVATE
        benchmark::benchmark
        benchmark::benchmark_main
        dino_lib
        MPI::MPI_CXX
    )
    target_include_directories(bm_Odyssey_L2Square
        PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../lib
        ${CMAKE_CURRENT_SOURCE_DIR}/../commons
    )
endif()

# ////// SING (conditional on CUDA) //////
if(SING_CUDA_AVAILABLE)
    if(DEBUG_MSG)
        message(STATUS "---")
        message(STATUS "## Benchmark: Sing L2Square")
    endif()

    add_executable(bm_Sing_L2Square
        bm_Sing_L2Square.cpp
        bm_utils.cpp
        ../commons/paramSetup.cpp
        ../commons/test_bm_utils.cpp
        ../commons/dataloaders.cpp
    )
    target_link_libraries(bm_Sing_L2Square
        PRIVATE
        benchmark::benchmark
        benchmark::benchmark_main
        dino_lib
    )
    target_include_directories(bm_Sing_L2Square
        PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../lib
        ${CMAKE_CURRENT_SOURCE_DIR}/../commons
    )
endif()

if(DEBUG_MSG)
    message(STATUS "--- Benchmark Configuration Complete ---")
endif()