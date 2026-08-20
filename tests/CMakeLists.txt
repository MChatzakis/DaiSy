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

# # ////// BRUTEFORCE DTW ////// 
#  if(DEBUG_MSG)
#      message(STATUS "---")
#      message(STATUS "Attempting to add executable: test_bruteforce_DTW")
#  endif()
#  add_executable(
#      test_bruteforce_DTW
#      test_bruteforce_DTW.cpp
#      test_utils.cpp
#  )
#  if(DEBUG_MSG)
#      message(STATUS "Executable test_bruteforce_DTW added.")
#  endif()
#
#  if(DEBUG_MSG)
#      message(STATUS "Linking libraries for test_bruteforce_DTW...")
#  endif()
#  target_link_libraries(
#      test_bruteforce_DTW
#      PRIVATE
#      GTest::gtest_main
#      dino_lib
#      commons_lib
#      stdc++fs
#  )
#  if(DEBUG_MSG)
#      message(STATUS "Libraries linked for test_bruteforce_DTW.")
#  endif()
#
#  if(DEBUG_MSG)
#      message(STATUS "Adding include directories for test_bruteforce_DTW...")
#  endif()
#  target_include_directories(test_bruteforce_DTW
#      PRIVATE
#      ${CMAKE_CURRENT_SOURCE_DIR}/../lib
#      ${CMAKE_CURRENT_SOURCE_DIR}/../commons
#  )
#  if(DEBUG_MSG)
#      message(STATUS "Include directories added for test_bruteforce_DTW.")
#  endif()
#
#  if(DEBUG_MSG)
#      message(STATUS "Discovering tests for test_bruteforce_DTW...")
#  endif()
#  gtest_discover_tests(
#      test_bruteforce_DTW
#      WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
#  )
#  if(DEBUG_MSG)
#      message(STATUS "Tests discovered for test_bruteforce_DTW.")
#  endif()

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

# # ////// LBBRUTEFORCE DTW ////// 
#  if(DEBUG_MSG)
#      message(STATUS "---")
#      message(STATUS "Attempting to add executable: test_LbBruteforce_DTW")
#  endif()
#  add_executable(
#      test_LbBruteforce_DTW
#      test_LbBruteforce_DTW.cpp
#      test_utils.cpp
#  )
#  if(DEBUG_MSG)
#      message(STATUS "Executable test_LbBruteforce_DTW added.")
#  endif()
#
#  if(DEBUG_MSG)
#      message(STATUS "Linking libraries for test_LbBruteforce_DTW...")
#  endif()
#  target_link_libraries(
#      test_LbBruteforce_DTW
#      PRIVATE
#      GTest::gtest_main
#      dino_lib
#      commons_lib
#      stdc++fs
#  )
#  if(DEBUG_MSG)
#      message(STATUS "Libraries linked for test_LbBruteforce_DTW.")
#  endif()
#
#  if(DEBUG_MSG)
#      message(STATUS "Adding include directories for test_LbBruteforce_DTW...")
#  endif()
#  target_include_directories(test_LbBruteforce_DTW
#      PRIVATE
#      ${CMAKE_CURRENT_SOURCE_DIR}/../lib
#      ${CMAKE_CURRENT_SOURCE_DIR}/../commons
#  )
#  if(DEBUG_MSG)
#      message(STATUS "Include directories added for test_LbBruteforce_DTW.")
#  endif()
#
#  if(DEBUG_MSG)
#      message(STATUS "Discovering tests for test_LbBruteforce_DTW...")
#  endif()
#  gtest_discover_tests(
#      test_LbBruteforce_DTW
#      WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
#  )
#  if(DEBUG_MSG)
#      message(STATUS "Tests discovered for test_LbBruteforce_DTW.")
#  endif()

# # ////// MESSI DTW ////// 
#  if(DEBUG_MSG)
#      message(STATUS "---")
#      message(STATUS "Attempting to add executable: test_Messi_DTW")
#  endif()
#  add_executable(
#      test_Messi_DTW
#      test_Messi_DTW.cpp
#      test_utils.cpp
#  )
#  if(DEBUG_MSG)
#      message(STATUS "Executable test_Messi_DTW added.")
#  endif()
#
#  if(DEBUG_MSG)
#      message(STATUS "Linking libraries for test_Messi_DTW...")
#  endif()
#  target_link_libraries(
#      test_Messi_DTW
#      PRIVATE
#      GTest::gtest_main
#      dino_lib
#      commons_lib
#      stdc++fs
#  )
#  if(DEBUG_MSG)
#      message(STATUS "Libraries linked for test_Messi_DTW.")
#  endif()
#
#  if(DEBUG_MSG)
#      message(STATUS "Adding include directories for test_Messi_DTW...")
#  endif()
#  target_include_directories(test_Messi_DTW
#      PRIVATE
#      ${CMAKE_CURRENT_SOURCE_DIR}/../lib
#      ${CMAKE_CURRENT_SOURCE_DIR}/../commons
#  )
#  if(DEBUG_MSG)
#      message(STATUS "Include directories added for test_Messi_DTW.")
#  endif()
#
#  if(DEBUG_MSG)
#      message(STATUS "Discovering tests for test_Messi_DTW...")
#  endif()
#  gtest_discover_tests(
#      test_Messi_DTW
#      WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
#  )
#  if(DEBUG_MSG)
#      message(STATUS "Tests discovered for test_Messi_DTW.")
#  endif()

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

# ////// ODYSSEY L2Square (conditional on MPI) //////
# Odyssey tests must run under mpirun (e.g. mpirun -np 4); one CTest entry runs all GTest cases.
if(DEBUG_MSG)
    message(STATUS "---")
endif()
if (BUILD_ODYSSEY_AVAILABLE)
    if(DEBUG_MSG)
        message(STATUS "BUILD_ODYSSEY_AVAILABLE is TRUE. Building test_Odyssey_L2Square.")
        message(STATUS "Attempting to add executable: test_Odyssey_L2Square")
    endif()
    add_executable(
        test_Odyssey_L2Square
        test_Odyssey_L2Square.cpp
        test_utils.cpp
    )
    if(DEBUG_MSG)
        message(STATUS "Executable test_Odyssey_L2Square added.")
    endif()

    target_compile_definitions(test_Odyssey_L2Square PRIVATE ODYSSEY_MPI)

    if(DEBUG_MSG)
        message(STATUS "Linking libraries for test_Odyssey_L2Square...")
    endif()
    target_link_libraries(
        test_Odyssey_L2Square
        PRIVATE
        GTest::gtest_main
        dino_lib
        commons_lib
        stdc++fs
        MPI::MPI_CXX
    )
    if(DEBUG_MSG)
        message(STATUS "Libraries linked for test_Odyssey_L2Square.")
    endif()

    if(DEBUG_MSG)
        message(STATUS "Adding include directories for test_Odyssey_L2Square...")
    endif()
    target_include_directories(test_Odyssey_L2Square
        PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../lib
        ${CMAKE_CURRENT_SOURCE_DIR}/../commons
    )
    if(DEBUG_MSG)
        message(STATUS "Include directories added for test_Odyssey_L2Square.")
    endif()

    # Run all Odyssey L2Square tests under mpirun (single CTest entry; GTest runs all parameterized cases)
    ## --oversubscribe allows 4 processes on runners with fewer cores (e.g. GitHub Actions)
    add_test(NAME test_Odyssey_L2Square COMMAND ${MPIEXEC_EXECUTABLE} --oversubscribe ${MPIEXEC_NUMPROC_FLAG} 4 $<TARGET_FILE:test_Odyssey_L2Square>)
    set_tests_properties(test_Odyssey_L2Square PROPERTIES WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        ENVIRONMENT "PMIX_MCA_pcompress_base_silence_warning=1")
    if(DEBUG_MSG)
        message(STATUS "Test test_Odyssey_L2Square added (run with mpirun -np 4).")
    endif()

    # # Odyssey DTW (same pattern: build + link + run under mpirun)
    # if(DEBUG_MSG)
    #     message(STATUS "Attempting to add executable: test_Odyssey_DTW")
    # endif()
    # add_executable(
    #     test_Odyssey_DTW
    #     test_Odyssey_DTW.cpp
    #     test_utils.cpp
    # )
    # target_compile_definitions(test_Odyssey_DTW PRIVATE ODYSSEY_MPI)
    # target_link_libraries(test_Odyssey_DTW PRIVATE GTest::gtest_main dino_lib commons_lib stdc++fs MPI::MPI_CXX)
    # target_include_directories(test_Odyssey_DTW PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../lib ${CMAKE_CURRENT_SOURCE_DIR}/../commons)
    # add_test(NAME test_Odyssey_DTW COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 4 $<TARGET_FILE:test_Odyssey_DTW>)
    # set_tests_properties(test_Odyssey_DTW PROPERTIES WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    #     ENVIRONMENT "PMIX_MCA_pcompress_base_silence_warning=1")
    # if(DEBUG_MSG)
    #     message(STATUS "Test test_Odyssey_DTW added (run with mpirun -np 4).")
    # endif()
else()
    if(DEBUG_MSG)
        message(STATUS "BUILD_ODYSSEY_AVAILABLE is FALSE. test_Odyssey_L2Square / test_Odyssey_DTW (MPI-dependent) will NOT be built.")
    endif()
endif()

# ////// PARIS L2Square //////
if(DEBUG_MSG)
    message(STATUS "---")
    message(STATUS "Attempting to add executable: test_ParIS_L2Square")
endif()
add_executable(
    test_ParIS_L2Square
    test_Paris_L2Square.cpp
    test_utils.cpp
)
if(DEBUG_MSG)
    message(STATUS "Executable test_ParIS_L2Square added.")
endif()

if(DEBUG_MSG)
    message(STATUS "Linking libraries for test_ParIS_L2Square...")
endif()
target_link_libraries(
    test_ParIS_L2Square
    PRIVATE
    GTest::gtest_main
    dino_lib
    commons_lib
    stdc++fs
)
if(DEBUG_MSG)
    message(STATUS "Libraries linked for test_ParIS_L2Square.")
endif()

if(DEBUG_MSG)
    message(STATUS "Adding include directories for test_ParIS_L2Square...")
endif()
target_include_directories(test_ParIS_L2Square
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../lib
    ${CMAKE_CURRENT_SOURCE_DIR}/../commons
)
if(DEBUG_MSG)
    message(STATUS "Include directories added for test_ParIS_L2Square.")
endif()

if(DEBUG_MSG)
    message(STATUS "Discovering tests for test_ParIS_L2Square...")
endif()
gtest_discover_tests(
    test_ParIS_L2Square
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
)
if(DEBUG_MSG)
    message(STATUS "Tests discovered for test_ParIS_L2Square.")
endif()

# ////// SING L2Square (conditional on CUDA) //////
if(DEBUG_MSG)
    message(STATUS "---")
    message(STATUS "## Test: Sing L2Square (conditional on CUDA)")
endif()

if(SING_CUDA_AVAILABLE)
    if(DEBUG_MSG)
        message(STATUS "SING_CUDA_AVAILABLE is TRUE. Building test_Sing_L2Square.")
        message(STATUS "Attempting to add executable: test_Sing_L2Square")
    endif()
    add_executable(
        test_Sing_L2Square
        test_Sing_L2Square.cpp
        test_utils.cpp
    )
    if(DEBUG_MSG)
        message(STATUS "Executable test_Sing_L2Square added.")
    endif()

    if(DEBUG_MSG)
        message(STATUS "Linking libraries for test_Sing_L2Square...")
    endif()
    target_link_libraries(
        test_Sing_L2Square
        PRIVATE
        GTest::gtest_main
        dino_lib
        commons_lib
        stdc++fs
        CUDA::cudart
    )
    if(DEBUG_MSG)
        message(STATUS "Libraries linked for test_Sing_L2Square.")
    endif()

    if(DEBUG_MSG)
        message(STATUS "Adding include directories for test_Sing_L2Square...")
    endif()
    target_include_directories(test_Sing_L2Square
        PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../lib
        ${CMAKE_CURRENT_SOURCE_DIR}/../commons
    )
    if(DEBUG_MSG)
        message(STATUS "Include directories added for test_Sing_L2Square.")
    endif()

    if(DEBUG_MSG)
        message(STATUS "Discovering tests for test_Sing_L2Square...")
    endif()
    gtest_discover_tests(
        test_Sing_L2Square
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    )
    if(DEBUG_MSG)
        message(STATUS "Tests discovered for test_Sing_L2Square.")
    endif()
else()
    if(DEBUG_MSG)
        message(STATUS "SING_CUDA_AVAILABLE is FALSE. test_Sing_L2Square (CUDA-dependent) will NOT be built.")
    endif()
endif()

# ////// SOFA L2Square (conditional on FFTW3) //////
if(BUILD_SOFA_AVAILABLE)
    if(DEBUG_MSG)
        message(STATUS "---")
        message(STATUS "BUILD_SOFA_AVAILABLE is TRUE. Building test_Sofa_L2Square.")
        message(STATUS "Attempting to add executable: test_Sofa_L2Square")
    endif()
    add_executable(
        test_Sofa_L2Square
        test_Sofa_L2Square.cpp
        test_utils.cpp
    )
    if(DEBUG_MSG)
        message(STATUS "Executable test_Sofa_L2Square added.")
    endif()

    if(DEBUG_MSG)
        message(STATUS "Linking libraries for test_Sofa_L2Square...")
    endif()
    target_link_libraries(
        test_Sofa_L2Square
        PRIVATE
        GTest::gtest_main
        dino_lib
        commons_lib
        stdc++fs
    )
    if(DEBUG_MSG)
        message(STATUS "Libraries linked for test_Sofa_L2Square.")
    endif()

    if(DEBUG_MSG)
        message(STATUS "Adding include directories for test_Sofa_L2Square...")
    endif()
    target_include_directories(test_Sofa_L2Square
        PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../lib
        ${CMAKE_CURRENT_SOURCE_DIR}/../commons
    )
    if(DEBUG_MSG)
        message(STATUS "Include directories added for test_Sofa_L2Square.")
    endif()

    if(DEBUG_MSG)
        message(STATUS "Discovering tests for test_Sofa_L2Square...")
    endif()
    gtest_discover_tests(
        test_Sofa_L2Square
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    )
    if(DEBUG_MSG)
        message(STATUS "Tests discovered for test_Sofa_L2Square.")
    endif()
else()
    if(DEBUG_MSG)
        message(STATUS "BUILD_SOFA_AVAILABLE is FALSE. test_Sofa_L2Square (FFTW3-dependent) will NOT be built.")
    endif()
endif()

# ////// HERCULES L2Square //////
add_executable(
    test_Hercules_L2Square
    test_Hercules_L2Square.cpp
    test_utils.cpp
)

target_link_libraries(
    test_Hercules_L2Square
    PRIVATE
    GTest::gtest_main
    dino_lib
    commons_lib
    stdc++fs
)

target_include_directories(test_Hercules_L2Square
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../lib
    ${CMAKE_CURRENT_SOURCE_DIR}/../commons
)

gtest_discover_tests(
    test_Hercules_L2Square
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
)

# ////// DUMPYOS L2Square //////
add_executable(
    test_DumpyOS_L2Square
    test_DumpyOS_L2Square.cpp
    test_utils.cpp
)

target_link_libraries(
    test_DumpyOS_L2Square
    PRIVATE
    GTest::gtest_main
    dino_lib
    commons_lib
    stdc++fs
)

target_include_directories(test_DumpyOS_L2Square
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../lib
    ${CMAKE_CURRENT_SOURCE_DIR}/../commons
)

gtest_discover_tests(
    test_DumpyOS_L2Square
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
)

# # ////// DUMPYOS DTW //////
# add_executable(
#     test_DumpyOS_DTW
#     test_DumpyOS_DTW.cpp
#     test_utils.cpp
# )
#
# target_link_libraries(
#     test_DumpyOS_DTW
#     PRIVATE
#     GTest::gtest_main
#     dino_lib
#     commons_lib
#     stdc++fs
# )
#
# target_include_directories(test_DumpyOS_DTW
#     PRIVATE
#     ${CMAKE_CURRENT_SOURCE_DIR}/../lib
#     ${CMAKE_CURRENT_SOURCE_DIR}/../commons
# )
#
# gtest_discover_tests(
#     test_DumpyOS_DTW
#     WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
# )

# ////// FRESH L2Square //////
add_executable(
    test_Fresh_L2Square
    test_Fresh_L2Square.cpp
    test_utils.cpp
)

target_link_libraries(
    test_Fresh_L2Square
    PRIVATE
    GTest::gtest_main
    dino_lib
    commons_lib
    stdc++fs
)

target_include_directories(test_Fresh_L2Square
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../lib
    ${CMAKE_CURRENT_SOURCE_DIR}/../commons
)

gtest_discover_tests(
    test_Fresh_L2Square
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
)

# ////// FRESH DTW //////
# Disabled until the DaiSy-wide DTW ground truth convention is unified
# with the C++ dtwsimdPruned output. Matches the state of the other DTW tests.
# add_executable(
#     test_Fresh_DTW
#     test_Fresh_DTW.cpp
#     test_utils.cpp
# )
#
# target_link_libraries(
#     test_Fresh_DTW
#     PRIVATE
#     GTest::gtest_main
#     dino_lib
#     commons_lib
#     stdc++fs
# )
#
# target_include_directories(test_Fresh_DTW
#     PRIVATE
#     ${CMAKE_CURRENT_SOURCE_DIR}/../lib
#     ${CMAKE_CURRENT_SOURCE_DIR}/../commons
# )
#
# gtest_discover_tests(
#     test_Fresh_DTW
#     WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
# )

# ////// Python range search test (requires BUILD_PYTHON + faiss) //////
if(BUILD_PYTHON)
    find_package(Python3 QUIET COMPONENTS Interpreter)
    if(Python3_FOUND)
        add_test(
            NAME test_range_search_py
            COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test_range_search.py
            WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        )
        set_tests_properties(test_range_search_py PROPERTIES
            ENVIRONMENT "PYTHONPATH=${CMAKE_SOURCE_DIR}"
        )
    endif()
endif()

# ////// MESSI Range //////
add_executable(
    test_Messi_Range
    test_Messi_Range.cpp
    test_utils.cpp
)

target_link_libraries(
    test_Messi_Range
    PRIVATE
    GTest::gtest_main
    dino_lib
    commons_lib
    stdc++fs
)

target_include_directories(test_Messi_Range
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../lib
    ${CMAKE_CURRENT_SOURCE_DIR}/../commons
)

gtest_discover_tests(
    test_Messi_Range
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
)

# ////// FRESH Range //////
add_executable(
    test_Fresh_Range
    test_Fresh_Range.cpp
    test_utils.cpp
)

target_link_libraries(
    test_Fresh_Range
    PRIVATE
    GTest::gtest_main
    dino_lib
    commons_lib
    stdc++fs
)

target_include_directories(test_Fresh_Range
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../lib
    ${CMAKE_CURRENT_SOURCE_DIR}/../commons
)

gtest_discover_tests(
    test_Fresh_Range
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
)

# ////// DUMPYOS Range //////
add_executable(
    test_DumpyOS_Range
    test_DumpyOS_Range.cpp
    test_utils.cpp
)

target_link_libraries(
    test_DumpyOS_Range
    PRIVATE
    GTest::gtest_main
    dino_lib
    commons_lib
    stdc++fs
)

target_include_directories(test_DumpyOS_Range
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../lib
    ${CMAKE_CURRENT_SOURCE_DIR}/../commons
)

gtest_discover_tests(
    test_DumpyOS_Range
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
)

# ////// LBBRUTEFORCE Range //////
add_executable(
    test_LbBruteforce_Range
    test_LbBruteforce_Range.cpp
    test_utils.cpp
)

target_link_libraries(
    test_LbBruteforce_Range
    PRIVATE
    GTest::gtest_main
    dino_lib
    commons_lib
    stdc++fs
)

target_include_directories(test_LbBruteforce_Range
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../lib
    ${CMAKE_CURRENT_SOURCE_DIR}/../commons
)

gtest_discover_tests(
    test_LbBruteforce_Range
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
)

# ////// SING Range (conditional on CUDA) //////
if(SING_CUDA_AVAILABLE)
    add_executable(
        test_Sing_Range
        test_Sing_Range.cpp
        test_utils.cpp
    )

    target_link_libraries(
        test_Sing_Range
        PRIVATE
        GTest::gtest_main
        dino_lib
        commons_lib
        stdc++fs
        CUDA::cudart
    )

    target_include_directories(test_Sing_Range
        PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../lib
        ${CMAKE_CURRENT_SOURCE_DIR}/../commons
    )

    gtest_discover_tests(
        test_Sing_Range
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    )
endif()

# ////// HERCULES Range //////
add_executable(
    test_Hercules_Range
    test_Hercules_Range.cpp
    test_utils.cpp
)

target_link_libraries(
    test_Hercules_Range
    PRIVATE
    GTest::gtest_main
    dino_lib
    commons_lib
    stdc++fs
)

target_include_directories(test_Hercules_Range
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../lib
    ${CMAKE_CURRENT_SOURCE_DIR}/../commons
)

gtest_discover_tests(
    test_Hercules_Range
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
)

# ////// PARIS Range //////
add_executable(
    test_ParIS_Range
    test_ParIS_Range.cpp
    test_utils.cpp
)

target_link_libraries(
    test_ParIS_Range
    PRIVATE
    GTest::gtest_main
    dino_lib
    commons_lib
    stdc++fs
)

target_include_directories(test_ParIS_Range
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../lib
    ${CMAKE_CURRENT_SOURCE_DIR}/../commons
)

gtest_discover_tests(
    test_ParIS_Range
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
)

# ////// SOFA Range (conditional on FFTW3) //////
if(BUILD_SOFA_AVAILABLE)
    add_executable(
        test_Sofa_Range
        test_Sofa_Range.cpp
        test_utils.cpp
    )

    target_link_libraries(
        test_Sofa_Range
        PRIVATE
        GTest::gtest_main
        dino_lib
        commons_lib
        stdc++fs
    )

    target_include_directories(test_Sofa_Range
        PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../lib
        ${CMAKE_CURRENT_SOURCE_DIR}/../commons
    )

    gtest_discover_tests(
        test_Sofa_Range
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    )
endif()

# ////// ODYSSEY Range (conditional on MPI) //////
if(BUILD_ODYSSEY_AVAILABLE)
    add_executable(
        test_Odyssey_Range
        test_Odyssey_Range.cpp
        test_utils.cpp
    )

    target_compile_definitions(test_Odyssey_Range PRIVATE ODYSSEY_MPI)

    target_link_libraries(
        test_Odyssey_Range
        PRIVATE
        GTest::gtest_main
        dino_lib
        commons_lib
        stdc++fs
        MPI::MPI_CXX
    )

    target_include_directories(test_Odyssey_Range
        PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../lib
        ${CMAKE_CURRENT_SOURCE_DIR}/../commons
    )

    add_test(NAME test_Odyssey_Range
        COMMAND ${MPIEXEC_EXECUTABLE} --oversubscribe ${MPIEXEC_NUMPROC_FLAG} 4
                $<TARGET_FILE:test_Odyssey_Range>)
    set_tests_properties(test_Odyssey_Range PROPERTIES
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        ENVIRONMENT "PMIX_MCA_pcompress_base_silence_warning=1")
endif()
# ////// COCONUT (static kNN) //////
if(BUILD_COCONUT)
    add_executable(test_Coconut_L2Square test_Coconut_L2Square.cpp test_utils.cpp)
    target_link_libraries(test_Coconut_L2Square PRIVATE GTest::gtest_main dino_lib commons_lib stdc++fs)
    target_include_directories(test_Coconut_L2Square PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../lib ${CMAKE_CURRENT_SOURCE_DIR}/../commons)
    gtest_discover_tests(test_Coconut_L2Square WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

    # ////// COCONUT (streaming kNN) //////
    add_executable(test_Coconut_Streaming test_Coconut_Streaming.cpp test_utils.cpp)
    target_link_libraries(test_Coconut_Streaming PRIVATE GTest::gtest_main dino_lib commons_lib stdc++fs)
    target_include_directories(test_Coconut_Streaming PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../lib ${CMAKE_CURRENT_SOURCE_DIR}/../commons)
    gtest_discover_tests(test_Coconut_Streaming WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

    # ////// COCONUT Range //////
    add_executable(test_Coconut_Range test_Coconut_Range.cpp test_utils.cpp)
    target_link_libraries(test_Coconut_Range PRIVATE GTest::gtest_main dino_lib commons_lib stdc++fs)
    target_include_directories(test_Coconut_Range PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../lib ${CMAKE_CURRENT_SOURCE_DIR}/../commons)
    gtest_discover_tests(test_Coconut_Range WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
elseif(DEBUG_MSG)
    message(STATUS "BUILD_COCONUT is OFF. Skipping Coconut tests.")
endif()

# non-normalized data
add_executable(
    test_equidepth
    test_equidepth.cpp
    test_utils.cpp
)
target_link_libraries(
    test_equidepth
    PRIVATE
    GTest::gtest_main
    dino_lib
    commons_lib
    stdc++fs
)
target_include_directories(test_equidepth
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../lib
    ${CMAKE_CURRENT_SOURCE_DIR}/../commons
)
gtest_discover_tests(
    test_equidepth
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
)
