find_package(OpenMP REQUIRED) 

target_sources(dino_lib 
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/BruteforceSearch.cpp
)

target_sources(dino_lib 
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/LbBruteforce.cpp
)

target_sources(dino_lib 
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/Messi.cpp
)

target_sources(dino_lib 
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/Odyssey.cpp
)

target_sources(dino_lib 
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/Paris.cpp
)

target_sources(dino_lib 
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/Sing.cpp
)

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