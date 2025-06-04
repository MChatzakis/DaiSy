if(BUILD_DEMO)
    # demo_bruteforce
    add_executable(demo_bruteforce demo_bruteforce.cpp)

    target_link_libraries(demo_bruteforce PRIVATE dino_lib commons_lib)

    target_include_directories(demo_bruteforce PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../lib
        ${CMAKE_CURRENT_SOURCE_DIR}/../commons
    )

    # demo_LbBruteforce
    add_executable(demo_LbBruteforce demo_LbBruteforce.cpp)

    target_link_libraries(demo_LbBruteforce PRIVATE dino_lib commons_lib)

    target_include_directories(demo_LbBruteforce PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../lib
        ${CMAKE_CURRENT_SOURCE_DIR}/../commons
    )    
endif()
