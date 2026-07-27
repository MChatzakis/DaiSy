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

FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG v1.14.0
)
set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

if(DEBUG_MSG)
    message(STATUS "GoogleTest fetched and available.")
endif()

# ====== FAISS (CPU-only) for benchmarks ======
set(FAISS_ENABLE_GPU OFF CACHE BOOL "" FORCE)
set(FAISS_ENABLE_PYTHON OFF CACHE BOOL "" FORCE)
set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
set(FAISS_ENABLE_EXTRAS OFF CACHE BOOL "" FORCE)
set(FAISS_ENABLE_C_API OFF CACHE BOOL "" FORCE)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/faiss ${CMAKE_CURRENT_BINARY_DIR}/faiss)

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

# ////// COCONUT //////
if(BUILD_COCONUT)
    add_executable(bm_Coconut_L2Square
        bm_Coconut_L2Square.cpp
        bm_utils.cpp
        ../commons/paramSetup.cpp
        ../commons/test_bm_utils.cpp
        ../commons/dataloaders.cpp
    )
    target_link_libraries(bm_Coconut_L2Square
        PRIVATE
        benchmark::benchmark
        benchmark::benchmark_main
        dino_lib
    )
    target_include_directories(bm_Coconut_L2Square
        PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../lib
        ${CMAKE_CURRENT_SOURCE_DIR}/../commons
    )
elseif(DEBUG_MSG)
    message(STATUS "BUILD_COCONUT is OFF. Skipping Coconut benchmarks.")
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

# ////// FAISS FLAT //////
if(DEBUG_MSG)
    message(STATUS "---")
    message(STATUS "## Benchmark: FAISS Flat L2Square")
endif()

add_executable(bm_FaissFlat_L2Square
    bm_FaissFlat_L2Square.cpp
    bm_utils.cpp
    ../commons/paramSetup.cpp
    ../commons/test_bm_utils.cpp
    ../commons/dataloaders.cpp
)

target_link_libraries(bm_FaissFlat_L2Square
    PRIVATE
    benchmark::benchmark
    benchmark::benchmark_main
    dino_lib
    faiss
)

target_include_directories(bm_FaissFlat_L2Square
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../lib
    ${CMAKE_CURRENT_SOURCE_DIR}/../commons
)

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

# ////// SOFA (conditional on FFTW3) //////
if(BUILD_SOFA_AVAILABLE)
add_executable(bm_Sofa_L2Square
    bm_Sofa_L2Square.cpp
    bm_utils.cpp
    ../commons/paramSetup.cpp
    ../commons/test_bm_utils.cpp
    ../commons/dataloaders.cpp
)
target_link_libraries(bm_Sofa_L2Square
    PRIVATE
    benchmark::benchmark
    benchmark::benchmark_main
    dino_lib
    GTest::gtest
)
target_include_directories(bm_Sofa_L2Square
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../lib
    ${CMAKE_CURRENT_SOURCE_DIR}/../commons
)
endif()

# ////// DUMPYOS //////
add_executable(bm_DumpyOS_L2Square
    bm_DumpyOS_L2Square.cpp
    bm_utils.cpp
    ../commons/paramSetup.cpp
    ../commons/test_bm_utils.cpp
    ../commons/dataloaders.cpp
)
target_link_libraries(bm_DumpyOS_L2Square
    PRIVATE
    benchmark::benchmark
    benchmark::benchmark_main
    dino_lib
    GTest::gtest
)
target_include_directories(bm_DumpyOS_L2Square
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../lib
    ${CMAKE_CURRENT_SOURCE_DIR}/../commons
)

# ////// HERCULES //////
add_executable(bm_Hercules_L2Square
    bm_Hercules_L2Square.cpp
    bm_utils.cpp
    ../commons/paramSetup.cpp
    ../commons/test_bm_utils.cpp
    ../commons/dataloaders.cpp
)
target_link_libraries(bm_Hercules_L2Square
    PRIVATE
    benchmark::benchmark
    benchmark::benchmark_main
    dino_lib
    GTest::gtest
)
target_include_directories(bm_Hercules_L2Square
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../lib
    ${CMAKE_CURRENT_SOURCE_DIR}/../commons
)

# ////// FRESH //////
add_executable(bm_Fresh_L2Square
    bm_Fresh_L2Square.cpp
    bm_utils.cpp
    ../commons/paramSetup.cpp
    ../commons/test_bm_utils.cpp
    ../commons/dataloaders.cpp
)
target_link_libraries(bm_Fresh_L2Square
    PRIVATE
    benchmark::benchmark
    benchmark::benchmark_main
    dino_lib
    GTest::gtest
)
target_include_directories(bm_Fresh_L2Square
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../lib
    ${CMAKE_CURRENT_SOURCE_DIR}/../commons
)

# ////// ODYSSEY (conditional on MPI) //////
if(BUILD_ODYSSEY_AVAILABLE)
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
if(BUILD_SING_AVAILABLE)
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

# ////// BRUTEFORCE RANGE //////
add_executable(bm_bruteforce_Range
    bm_bruteforce_Range.cpp
    bm_utils.cpp
    ../commons/paramSetup.cpp
    ../commons/test_bm_utils.cpp
    ../commons/dataloaders.cpp
)
target_link_libraries(bm_bruteforce_Range
    PRIVATE
    benchmark::benchmark
    benchmark::benchmark_main
    dino_lib
)
target_include_directories(bm_bruteforce_Range
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../lib
    ${CMAKE_CURRENT_SOURCE_DIR}/../commons
)

# ////// LBBRUTEFORCE RANGE //////
add_executable(bm_LbBruteforce_Range
    bm_LbBruteforce_Range.cpp
    bm_utils.cpp
    ../commons/paramSetup.cpp
    ../commons/test_bm_utils.cpp
    ../commons/dataloaders.cpp
)
target_link_libraries(bm_LbBruteforce_Range
    PRIVATE
    benchmark::benchmark
    benchmark::benchmark_main
    dino_lib
)
target_include_directories(bm_LbBruteforce_Range
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../lib
    ${CMAKE_CURRENT_SOURCE_DIR}/../commons
)

# ////// MESSI RANGE //////
add_executable(bm_Messi_Range
    bm_Messi_Range.cpp
    bm_utils.cpp
    ../commons/paramSetup.cpp
    ../commons/test_bm_utils.cpp
    ../commons/dataloaders.cpp
)
target_link_libraries(bm_Messi_Range
    PRIVATE
    benchmark::benchmark
    benchmark::benchmark_main
    dino_lib
)
target_include_directories(bm_Messi_Range
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../lib
    ${CMAKE_CURRENT_SOURCE_DIR}/../commons
)

# ////// FRESH RANGE //////
add_executable(bm_Fresh_Range
    bm_Fresh_Range.cpp
    bm_utils.cpp
    ../commons/paramSetup.cpp
    ../commons/test_bm_utils.cpp
    ../commons/dataloaders.cpp
)
target_link_libraries(bm_Fresh_Range
    PRIVATE
    benchmark::benchmark
    benchmark::benchmark_main
    dino_lib
)
target_include_directories(bm_Fresh_Range
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../lib
    ${CMAKE_CURRENT_SOURCE_DIR}/../commons
)

# ////// DUMPYOS RANGE //////
add_executable(bm_DumpyOS_Range
    bm_DumpyOS_Range.cpp
    bm_utils.cpp
    ../commons/paramSetup.cpp
    ../commons/test_bm_utils.cpp
    ../commons/dataloaders.cpp
)
target_link_libraries(bm_DumpyOS_Range
    PRIVATE
    benchmark::benchmark
    benchmark::benchmark_main
    dino_lib
)
target_include_directories(bm_DumpyOS_Range
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../lib
    ${CMAKE_CURRENT_SOURCE_DIR}/../commons
)

# ////// HERCULES RANGE //////
add_executable(bm_Hercules_Range
    bm_Hercules_Range.cpp
    bm_utils.cpp
    ../commons/paramSetup.cpp
    ../commons/test_bm_utils.cpp
    ../commons/dataloaders.cpp
)
target_link_libraries(bm_Hercules_Range
    PRIVATE
    benchmark::benchmark
    benchmark::benchmark_main
    dino_lib
)
target_include_directories(bm_Hercules_Range
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../lib
    ${CMAKE_CURRENT_SOURCE_DIR}/../commons
)

# ////// PARIS RANGE //////
add_executable(bm_ParIS_Range
    bm_ParIS_Range.cpp
    bm_utils.cpp
    ../commons/paramSetup.cpp
    ../commons/test_bm_utils.cpp
    ../commons/dataloaders.cpp
)
target_link_libraries(bm_ParIS_Range
    PRIVATE
    benchmark::benchmark
    benchmark::benchmark_main
    dino_lib
)
target_include_directories(bm_ParIS_Range
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../lib
    ${CMAKE_CURRENT_SOURCE_DIR}/../commons
)

# ////// SOFA RANGE (conditional on FFTW3) //////
if(BUILD_SOFA_AVAILABLE)
add_executable(bm_Sofa_Range
    bm_Sofa_Range.cpp
    bm_utils.cpp
    ../commons/paramSetup.cpp
    ../commons/test_bm_utils.cpp
    ../commons/dataloaders.cpp
)
target_link_libraries(bm_Sofa_Range
    PRIVATE
    benchmark::benchmark
    benchmark::benchmark_main
    dino_lib
)
target_include_directories(bm_Sofa_Range
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../lib
    ${CMAKE_CURRENT_SOURCE_DIR}/../commons
)
endif()

# ////// ODYSSEY RANGE (conditional on MPI) //////
if(BUILD_ODYSSEY_AVAILABLE)
add_executable(bm_Odyssey_Range
    bm_Odyssey_Range.cpp
    bm_utils.cpp
    ../commons/paramSetup.cpp
    ../commons/test_bm_utils.cpp
    ../commons/dataloaders.cpp
)
target_compile_definitions(bm_Odyssey_Range PRIVATE ODYSSEY_MPI=1)
target_link_libraries(bm_Odyssey_Range
    PRIVATE
    benchmark::benchmark
    benchmark::benchmark_main
    dino_lib
    MPI::MPI_CXX
)
target_include_directories(bm_Odyssey_Range
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../lib
    ${CMAKE_CURRENT_SOURCE_DIR}/../commons
)
endif()

# ////// SING RANGE (conditional on CUDA) //////
if(BUILD_SING_AVAILABLE)
add_executable(bm_Sing_Range
    bm_Sing_Range.cpp
    bm_utils.cpp
    ../commons/paramSetup.cpp
    ../commons/test_bm_utils.cpp
    ../commons/dataloaders.cpp
)
target_link_libraries(bm_Sing_Range
    PRIVATE
    benchmark::benchmark
    benchmark::benchmark_main
    dino_lib
)
target_include_directories(bm_Sing_Range
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../lib
    ${CMAKE_CURRENT_SOURCE_DIR}/../commons
)
endif()

if(DEBUG_MSG)
    message(STATUS "--- Benchmark Configuration Complete ---")
endif()
