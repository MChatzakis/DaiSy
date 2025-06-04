find_package(OpenMP REQUIRED) 

target_sources(dino_lib 
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/BruteforceSearch.cpp
)

target_sources(dino_lib 
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/LbBruteforceSearch.cpp
)

# target_sources(dino_lib 
#     PRIVATE
#     ${CMAKE_CURRENT_SOURCE_DIR}/MessiSearch.cpp
# )

# target_sources(dino_lib 
#     PRIVATE
#     ${CMAKE_CURRENT_SOURCE_DIR}/OdysseySearch.cpp
# )

# target_sources(dino_lib 
#     PRIVATE
#     ${CMAKE_CURRENT_SOURCE_DIR}/ParisSearch.cpp
# )

# target_sources(dino_lib 
#     PRIVATE
#     ${CMAKE_CURRENT_SOURCE_DIR}/SingSearch.cpp
# )

if(OpenMP_CXX_FOUND)
    target_compile_options(
        dino_lib 
        PRIVATE 
        ${OpenMP_CXX_FLAGS}
    )

    target_link_libraries(
        dino_lib 
        PRIVATE 
        OpenMP::OpenMP_CXX
    )
    
else()
    message(WARNING "OpenMP not found. Proceeding without OpenMP support.")
endif()