include(FetchContent)

FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
)

# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

enable_testing()

FetchContent_MakeAvailable(googletest)

include(GoogleTest)

# Test: BruteForce L2Square
add_executable(
    test_bruteforce_L2Square 
    test_bruteforce_L2Square.cpp
    test_utils.cpp
)

target_link_libraries(
    test_bruteforce_L2Square
    PRIVATE
    GTest::gtest_main
    dino_lib
    commons_lib
    stdc++fs  
)

target_include_directories(test_bruteforce_L2Square 
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../lib
    ${CMAKE_CURRENT_SOURCE_DIR}/../commons
)

# Test: LbBruteforce L2Square
add_executable(
    test_LbBruteforce_L2Square 
    test_LbBruteforce_L2Square.cpp
    test_utils.cpp
)

target_link_libraries(
    test_LbBruteforce_L2Square
    PRIVATE
    GTest::gtest_main
    dino_lib
    commons_lib
    stdc++fs  
)

target_include_directories(test_LbBruteforce_L2Square 
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../lib
    ${CMAKE_CURRENT_SOURCE_DIR}/../commons
)

# # Test: Messi L2Square
# add_executable(
#     test_Messi_L2Square 
#     test_Messi_L2Square.cpp
#     test_utils.cpp
# )

# target_link_libraries(
#     test_Messi_L2Square
#     PRIVATE
#     GTest::gtest_main
#     dino_lib
#     commons_lib
#     stdc++fs  
# )

# target_include_directories(test_Messi_L2Square 
#     PRIVATE
#     ${CMAKE_CURRENT_SOURCE_DIR}/../lib
#     ${CMAKE_CURRENT_SOURCE_DIR}/../commons
# )

# # Test: Odyssey L2Square 
# add_executable(
#     test_Odyssey_L2Square
#     test_Odyssey_L2Square.cpp
#     test_utils.cpp
# )

# target_link_libraries(
#     test_Odyssey_L2Square
#     PRIVATE
#     GTest::gtest_main
#     dino_lib
#     commons_lib
#     stdc++fs
# )

# target_include_directories(test_Odyssey_L2Square
#     PRIVATE
#     ${CMAKE_CURRENT_SOURCE_DIR}/../lib
#     ${CMAKE_CURRENT_SOURCE_DIR}/../commons
# )

# # Test: Paris L2Square 
# add_executable(
#     test_Paris_L2Square
#     test_Paris_L2Square.cpp
#     test_utils.cpp
# )

# target_link_libraries(
#     test_Paris_L2Square
#     PRIVATE
#     GTest::gtest_main
#     dino_lib
#     commons_lib
#     stdc++fs
# )

# target_include_directories(test_Paris_L2Square
#     PRIVATE
#     ${CMAKE_CURRENT_SOURCE_DIR}/../lib
#     ${CMAKE_CURRENT_SOURCE_DIR}/../commons
# )

# # Test: Sing L2Square 
# add_executable(
#     test_Sing_L2Square
#     test_Sing_L2Square.cpp
#     test_utils.cpp
# )

# target_link_libraries(
#     test_Sing_L2Square
#     PRIVATE
#     GTest::gtest_main
#     dino_lib
#     commons_lib
#     stdc++fs
# )

# target_include_directories(test_Sing_L2Square
#     PRIVATE
#     ${CMAKE_CURRENT_SOURCE_DIR}/../lib
#     ${CMAKE_CURRENT_SOURCE_DIR}/../commons
# )

# Discover and register all tests
gtest_discover_tests(
    test_bruteforce_L2Square
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
)

gtest_discover_tests(
    test_LbBruteforce_L2Square
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
)

# gtest_discover_tests(
#     test_Messi_L2Square
#     WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
# )

# gtest_discover_tests(
#     test_Odyssey_L2Square
#     WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
# )

# gtest_discover_tests(
#     test_Paris_L2Square
#     WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
# )

# gtest_discover_tests(
#     test_Sing_L2Square
#     WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
# )