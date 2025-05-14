add_executable(test_bruteforce test_bruteforce.cpp)

target_link_libraries(test_bruteforce PRIVATE dino_lib commons_lib)

target_include_directories(test_bruteforce PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../lib
    ${CMAKE_CURRENT_SOURCE_DIR}/../commons
)