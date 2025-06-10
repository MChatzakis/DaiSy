if(NOT BUILD_BENCHMARK)
    return()
endif()

include(FetchContent)

FetchContent_Declare(
  benchmark
  GIT_REPOSITORY https://github.com/google/benchmark.git
  GIT_TAG v1.8.3
)

FetchContent_MakeAvailable(benchmark)

# BruteForce
add_executable(bm_bruteforce_L2Square 
    bm_bruteforce_L2Square.cpp
    bm_utils.cpp
    ../commons/paramSetup.cpp
    ../commons/test_bm_utils.cpp
    ../commons/dataloaders.cpp
)

target_link_libraries(bm_bruteforce_L2Square 
    PRIVATE 
    benchmark::benchmark 
    benchmark::benchmark_main 
    dino_lib
)

target_include_directories(bm_bruteforce_L2Square 
    PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}/../lib
    ${CMAKE_CURRENT_SOURCE_DIR}/../commons
)

#LbBruteforce
add_executable(bm_LbBruteforce_L2Square 
    bm_LbBruteforce_L2Square.cpp
    bm_utils.cpp
    ../commons/paramSetup.cpp
    ../commons/test_bm_utils.cpp
    ../commons/dataloaders.cpp
)

target_link_libraries(bm_LbBruteforce_L2Square 
    PRIVATE 
    benchmark::benchmark 
    benchmark::benchmark_main 
    dino_lib
)

target_include_directories(bm_LbBruteforce_L2Square 
    PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}/../lib
    ${CMAKE_CURRENT_SOURCE_DIR}/../commons
)

# Odyssey Benchmark (conditional on MPI)
# if (ODYSSEY_MPI_AVAILABLE)
#     add_executable(bm_Odyssey_L2Square
#         bm_Odyssey_L2Square.cpp
#         bm_utils.cpp
#         ../commons/paramSetup.cpp
#         ../commons/test_bm_utils.cpp
#         ../commons/dataloaders.cpp
#     )

#     target_link_libraries(bm_Odyssey_L2Square
#         PRIVATE
#         benchmark::benchmark
#         benchmark::benchmark_main
#         dino_lib
#         MPI::MPI_CXX # Link with MPI for this benchmark
#     )

#     target_include_directories(bm_Odyssey_L2Square
#         PRIVATE
#         ${CMAKE_CURRENT_SOURCE_DIR}/../lib
#         ${CMAKE_CURRENT_SOURCE_DIR}/../commons
#     )
# else()
#     message(STATUS "bm_Odyssey_L2Square (MPI-dependent) will NOT be built.")
# endif()