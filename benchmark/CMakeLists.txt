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
