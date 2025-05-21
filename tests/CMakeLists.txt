include(FetchContent)

FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
)

# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

enable_testing()

FetchContent_MakeAvailable(googletest)

add_executable(
    test_bruteforce 
    test_bruteforce.cpp
    test_utils.cpp
)

target_link_libraries(
    test_bruteforce
    PRIVATE
    GTest::gtest_main
    dino_lib
    commons_lib
    stdc++fs  
)

target_include_directories(test_bruteforce 
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../lib
    ${CMAKE_CURRENT_SOURCE_DIR}/../commons
)

include(GoogleTest)

# to discover and register all tests in the executable
gtest_discover_tests(
    test_bruteforce
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
)