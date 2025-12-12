include(FetchContent)

if(DEBUG_MSG)
    message(STATUS "--- Configuring Test Builds ---")
endif()

FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
)

# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
if(DEBUG_MSG)
    message(STATUS "googletest: gtest_force_shared_crt set to ${gtest_force_shared_crt}")
endif()

enable_testing()
if(DEBUG_MSG)
    message(STATUS "Testing enabled.")
endif()

FetchContent_MakeAvailable(googletest)
if(DEBUG_MSG)
    message(STATUS "googletest made available.")
endif()

include(GoogleTest)
if(DEBUG_MSG)
    message(STATUS "GoogleTest module included.")
endif()

# ////// BRUTEFORCE L2Square //////
if(DEBUG_MSG)
    message(STATUS "---")
    message(STATUS "Attempting to add executable: test_bruteforce_L2Square")
endif()
add_executable(
    test_bruteforce_L2Square
    test_bruteforce_L2Square.cpp
    test_utils.cpp
)
if(DEBUG_MSG)
    message(STATUS "Executable test_bruteforce_L2Square added.")
endif()

if(DEBUG_MSG)
    message(STATUS "Linking libraries for test_bruteforce_L2Square...")
endif()
target_link_libraries(
    test_bruteforce_L2Square
    PRIVATE
    GTest::gtest_main
    dino_lib
    commons_lib
    stdc++fs
)
if(DEBUG_MSG)
    message(STATUS "Libraries linked for test_bruteforce_L2Square.")
endif()

if(DEBUG_MSG)
    message(STATUS "Adding include directories for test_bruteforce_L2Square...")
endif()
target_include_directories(test_bruteforce_L2Square
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../lib
    ${CMAKE_CURRENT_SOURCE_SITR}/../commons
)
if(DEBUG_MSG)
    message(STATUS "Include directories added for test_bruteforce_L2Square.")
endif()

if(DEBUG_MSG)
    message(STATUS "Discovering tests for test_bruteforce_L2Square...")
endif()
gtest_discover_tests(
    test_bruteforce_L2Square
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
)
if(DEBUG_MSG)
    message(STATUS "Tests discovered for test_bruteforce_L2Square.")
endif()

# ////// BRUTEFORCE DTW ////// 
 if(DEBUG_MSG)
     message(STATUS "---")
     message(STATUS "Attempting to add executable: test_bruteforce_DTW")
 endif()
 add_executable(
     test_bruteforce_DTW
     test_bruteforce_DTW.cpp
     test_utils.cpp
 )
 if(DEBUG_MSG)
     message(STATUS "Executable test_bruteforce_DTW added.")
 endif()

 if(DEBUG_MSG)
     message(STATUS "Linking libraries for test_bruteforce_DTW...")
 endif()
 target_link_libraries(
     test_bruteforce_DTW
     PRIVATE
     GTest::gtest_main
     dino_lib
     commons_lib
     stdc++fs
 )
 if(DEBUG_MSG)
     message(STATUS "Libraries linked for test_bruteforce_DTW.")
 endif()

 if(DEBUG_MSG)
     message(STATUS "Adding include directories for test_bruteforce_DTW...")
 endif()
 target_include_directories(test_bruteforce_DTW
     PRIVATE
     ${CMAKE_CURRENT_SOURCE_DIR}/../lib
     ${CMAKE_CURRENT_SOURCE_DIR}/../commons
 )
 if(DEBUG_MSG)
     message(STATUS "Include directories added for test_bruteforce_DTW.")
 endif()

 if(DEBUG_MSG)
     message(STATUS "Discovering tests for test_bruteforce_DTW...")
 endif()
 gtest_discover_tests(
     test_bruteforce_DTW
     WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
 )
 if(DEBUG_MSG)
     message(STATUS "Tests discovered for test_bruteforce_DTW.")
 endif()

# ////// LBBRUTEFORCE L2Square //////
if(DEBUG_MSG)
    message(STATUS "---")
    message(STATUS "Attempting to add executable: test_LbBruteforce_L2Square")
endif()
add_executable(
    test_LbBruteforce_L2Square
    test_LbBruteforce_L2Square.cpp
    test_utils.cpp
)
if(DEBUG_MSG)
    message(STATUS "Executable test_LbBruteforce_L2Square added.")
endif()

if(DEBUG_MSG)
    message(STATUS "Linking libraries for test_LbBruteforce_L2Square...")
endif()
target_link_libraries(
    test_LbBruteforce_L2Square
    PRIVATE
    GTest::gtest_main
    dino_lib
    commons_lib
    stdc++fs
)
if(DEBUG_MSG)
    message(STATUS "Libraries linked for test_LbBruteforce_L2Square.")
endif()

if(DEBUG_MSG)
    message(STATUS "Adding include directories for test_LbBruteforce_L2Square...")
endif()
target_include_directories(test_LbBruteforce_L2Square
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../lib
    ${CMAKE_CURRENT_SOURCE_DIR}/../commons
)
if(DEBUG_MSG)
    message(STATUS "Include directories added for test_LbBruteforce_L2Square.")
endif()

if(DEBUG_MSG)
    message(STATUS "Discovering tests for test_LbBruteforce_L2Square...")
endif()
gtest_discover_tests(
    test_LbBruteforce_L2Square
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
)
if(DEBUG_MSG)
    message(STATUS "Tests discovered for test_LbBruteforce_L2Square.")
endif()

 ////// LBBRUTEFORCE DTW ////// 
 if(DEBUG_MSG)
     message(STATUS "---")
     message(STATUS "Attempting to add executable: test_LbBruteforce_DTW")
 endif()
 add_executable(
     test_LbBruteforce_DTW
     test_LbBruteforce_DTW.cpp
     test_utils.cpp
 )
 if(DEBUG_MSG)
     message(STATUS "Executable test_LbBruteforce_DTW added.")
 endif()

 if(DEBUG_MSG)
     message(STATUS "Linking libraries for test_LbBruteforce_DTW...")
 endif()
 target_link_libraries(
     test_LbBruteforce_DTW
     PRIVATE
     GTest::gtest_main
     dino_lib
     commons_lib
     stdc++fs
 )
 if(DEBUG_MSG)
     message(STATUS "Libraries linked for test_LbBruteforce_DTW.")
 endif()

 if(DEBUG_MSG)
     message(STATUS "Adding include directories for test_LbBruteforce_DTW...")
 endif()
 target_include_directories(test_LbBruteforce_DTW
     PRIVATE
     ${CMAKE_CURRENT_SOURCE_DIR}/../lib
     ${CMAKE_CURRENT_SOURCE_DIR}/../commons
 )
 if(DEBUG_MSG)
     message(STATUS "Include directories added for test_LbBruteforce_DTW.")
 endif()

 if(DEBUG_MSG)
     message(STATUS "Discovering tests for test_LbBruteforce_DTW...")
 endif()
 gtest_discover_tests(
     test_LbBruteforce_DTW
     WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
 )
 if(DEBUG_MSG)
     message(STATUS "Tests discovered for test_LbBruteforce_DTW.")
 endif()

# ////// MESSI DTW ////// 
 if(DEBUG_MSG)
     message(STATUS "---")
     message(STATUS "Attempting to add executable: test_Messi_DTW")
 endif()
 add_executable(
     test_Messi_DTW
     test_Messi_DTW.cpp
     test_utils.cpp
 )
 if(DEBUG_MSG)
     message(STATUS "Executable test_Messi_DTW added.")
 endif()

 if(DEBUG_MSG)
     message(STATUS "Linking libraries for test_Messi_DTW...")
 endif()
 target_link_libraries(
     test_Messi_DTW
     PRIVATE
     GTest::gtest_main
     dino_lib
     commons_lib
     stdc++fs
 )
 if(DEBUG_MSG)
     message(STATUS "Libraries linked for test_Messi_DTW.")
 endif()

 if(DEBUG_MSG)
     message(STATUS "Adding include directories for test_Messi_DTW...")
 endif()
 target_include_directories(test_Messi_DTW
     PRIVATE
     ${CMAKE_CURRENT_SOURCE_DIR}/../lib
     ${CMAKE_CURRENT_SOURCE_DIR}/../commons
 )
 if(DEBUG_MSG)
     message(STATUS "Include directories added for test_Messi_DTW.")
 endif()

 if(DEBUG_MSG)
     message(STATUS "Discovering tests for test_Messi_DTW...")
 endif()
 gtest_discover_tests(
     test_Messi_DTW
     WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
 )
 if(DEBUG_MSG)
     message(STATUS "Tests discovered for test_Messi_DTW.")
 endif()

# ////// MESSI L2Square ////// 
 if(DEBUG_MSG)
     message(STATUS "---")
     message(STATUS "Attempting to add executable: test_Messi_L2Square")
 endif()
 add_executable(
     test_Messi_L2Square
     test_Messi_L2Square.cpp
     test_utils.cpp
 )
 if(DEBUG_MSG)
     message(STATUS "Executable test_Messi_L2Square added.")
 endif()

 if(DEBUG_MSG)
     message(STATUS "Linking libraries for test_Messi_L2Square...")
 endif()
 target_link_libraries(
     test_Messi_L2Square
     PRIVATE
     GTest::gtest_main
     dino_lib
     commons_lib
     stdc++fs
 )
 if(DEBUG_MSG)
     message(STATUS "Libraries linked for test_Messi_L2Square.")
 endif()

 if(DEBUG_MSG)
     message(STATUS "Adding include directories for test_Messi_L2Square...")
 endif()
 target_include_directories(test_Messi_L2Square
     PRIVATE
     ${CMAKE_CURRENT_SOURCE_DIR}/../lib
     ${CMAKE_CURRENT_SOURCE_DIR}/../commons
 )
 if(DEBUG_MSG)
     message(STATUS "Include directories added for test_Messi_L2Square.")
 endif()

 if(DEBUG_MSG)
     message(STATUS "Discovering tests for test_Messi_L2Square...")
 endif()
 gtest_discover_tests(
     test_Messi_L2Square
     WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
 )
 if(DEBUG_MSG)
     message(STATUS "Tests discovered for test_Messi_L2Square.")
 endif()

# ////// ODYSSEY L2Square (conditional on MPI) ////// (COMMENTED OUT - UNDER IMPLEMENTATION)
# if(DEBUG_MSG)
#     message(STATUS "---")
# endif()
# if (ODYSSEY_MPI_AVAILABLE)
#     if(DEBUG_MSG)
#         message(STATUS "ODYSSEY_MPI_AVAILABLE is TRUE. Building test_Odyssey_L2Square.")
#         message(STATUS "Attempting to add executable: test_Odyssey_L2Square")
#     endif()
#     add_executable(
#         test_Odyssey_L2Square
#         test_Odyssey_L2Square.cpp
#         test_utils.cpp
#     )
#     if(DEBUG_MSG)
#         message(STATUS "Executable test_Odyssey_L2Square added.")
#     endif()

#     if(DEBUG_MSG)
#         message(STATUS "Linking libraries for test_Odyssey_L2Square...")
#     endif()
#     target_link_libraries(
#         test_Odyssey_L2Square
#         PRIVATE
#         GTest::gtest_main
#         dino_lib
#         commons_lib
#         stdc++fs
#         MPI::MPI_CXX # Link with MPI for this test
#     )
#     if(DEBUG_MSG)
#         message(STATUS "Libraries linked for test_Odyssey_L2Square.")
#     endif()

#     if(DEBUG_MSG)
#         message(STATUS "Adding include directories for test_Odyssey_L2Square...")
#     endif()
#     target_include_directories(test_Odyssey_L2Square
#         PRIVATE
#         ${CMAKE_CURRENT_SOURCE_DIR}/../lib
#         ${CMAKE_CURRENT_SOURCE_DIR}/../commons
#     )
#     if(DEBUG_MSG)
#         message(STATUS "Include directories added for test_Odyssey_L2Square.")
#     endif()

#     if(DEBUG_MSG)
#         message(STATUS "Discovering tests for test_Odyssey_L2Square...")
#     endif()
#     gtest_discover_tests(
#         test_Odyssey_L2Square
#         WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
#     )
#     if(DEBUG_MSG)
#         message(STATUS "Tests discovered for test_Odyssey_L2Square.")
#     endif()
# else()
#     if(DEBUG_MSG)
#         message(STATUS "ODYSSEY_MPI_AVAILABLE is FALSE. test_Odyssey_L2Square (MPI-dependent) will NOT be built.")
#     endif()
# endif()

# ////// PARIS L2Square //////
# if(DEBUG_MSG)
#     message(STATUS "---")
#     message(STATUS "Attempting to add executable: test_ParIS_L2Square")
# endif()
# add_executable(
#     test_ParIS_L2Square
#     test_ParIS_L2Square.cpp
#     test_utils.cpp
# )
# if(DEBUG_MSG)
#     message(STATUS "Executable test_ParIS_L2Square added.")
# endif()

# if(DEBUG_MSG)
#     message(STATUS "Linking libraries for test_ParIS_L2Square...")
# endif()
# target_link_libraries(
#     test_ParIS_L2Square
#     PRIVATE
#     GTest::gtest_main
#     dino_lib
#     commons_lib
#     stdc++fs
# )
# if(DEBUG_MSG)
#     message(STATUS "Libraries linked for test_ParIS_L2Square.")
# endif()

# if(DEBUG_MSG)
#     message(STATUS "Adding include directories for test_ParIS_L2Square...")
# endif()
# target_include_directories(test_ParIS_L2Square
#     PRIVATE
#     ${CMAKE_CURRENT_SOURCE_DIR}/../lib
#     ${CMAKE_CURRENT_SOURCE_DIR}/../commons
# )
# if(DEBUG_MSG)
#     message(STATUS "Include directories added for test_ParIS_L2Square.")
# endif()

# if(DEBUG_MSG)
#     message(STATUS "Discovering tests for test_ParIS_L2Square...")
# endif()
# gtest_discover_tests(
#     test_ParIS_L2Square
#     WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
# )
# if(DEBUG_MSG)
#     message(STATUS "Tests discovered for test_ParIS_L2Square.")
# endif()

# ////// SING L2Square ////// (COMMENTED OUT - UNDER IMPLEMENTATION)
# if(DEBUG_MSG)
#     message(STATUS "---")
#     message(STATUS "## Test: Sing L2Square (conditional on CUDA)") # Updated message for clarity
# endif()

# if(SING_CUDA_AVAILABLE)
#     if(DEBUG_MSG)
#         message(STATUS "SING_CUDA_AVAILABLE is TRUE. Building test_Sing_L2Square.")
#         message(STATUS "Attempting to add executable: test_Sing_L2Square")
#     endif()
#     add_executable(
#         test_Sing_L2Square
#         test_Sing_L2Square.cpp
#         test_utils.cpp
#     )
#     if(DEBUG_MSG)
#         message(STATUS "Executable test_Sing_L2Square added.")
#     endif()

#     if(DEBUG_MSG)
#         message(STATUS "Linking libraries for test_Sing_L2Square...")
#     endif()
#     target_link_libraries(
#         test_Sing_L2Square
#         PRIVATE
#         GTest::gtest_main
#         dino_lib
#         commons_lib
#         stdc++fs
#     )
#     if(DEBUG_MSG)
#         message(STATUS "Libraries linked for test_Sing_L2Square.")
#     endif()

#     if(DEBUG_MSG)
#         message(STATUS "Adding include directories for test_Sing_L2Square...")
#     endif()
#     target_include_directories(test_Sing_L2Square
#         PRIVATE
#         ${CMAKE_CURRENT_SOURCE_DIR}/../lib
#         ${CMAKE_CURRENT_SOURCE_DIR}/../commons
#     )
#     if(DEBUG_MSG)
#         message(STATUS "Include directories added for test_Sing_L2Square.")
#     endif()

#     if(DEBUG_MSG)
#         message(STATUS "Discovering tests for test_Sing_L2Square...")
#     endif()
#     gtest_discover_tests(
#         test_Sing_L2Square
#         WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
#     )
#     if(DEBUG_MSG)
#         message(STATUS "Tests discovered for test_Sing_L2Square.")
#     endif()
# else()
#     if(DEBUG_MSG)
#         message(STATUS "SING_CUDA_AVAILABLE is FALSE. test_Sing_L2Square (CUDA-dependent) will NOT be built.")
#     endif()
# endif()