find_package(OpenMP REQUIRED) 

# ////// BRUTEFORCE //////
target_sources(dino_lib 
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/Bruteforce.cpp
)

# ////// LBBRUTEFORCE //////
target_sources(dino_lib 
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/LbBruteforce.cpp
)

# # ////// MESSI //////
# target_sources(dino_lib 
#     PRIVATE
#     ${CMAKE_CURRENT_SOURCE_DIR}/Messi.cpp
# )

# ////// ODYSSEY //////
# ONLY add MPI includes and link MPI if ODYSSEY_MPI_AVAILABLE is true
if (ODYSSEY_MPI_AVAILABLE)    
    target_include_directories(dino_lib PUBLIC
        ${MPI_C_INCLUDE_DIRS} 
    )
    
    target_link_libraries(dino_lib PUBLIC
        ${MPI_C_LIBRARIES} 
    )
endif()

# ////// PARIS //////
# target_sources(dino_lib 
#     PRIVATE
#     ${CMAKE_CURRENT_SOURCE_DIR}/ParIS.cpp
# )

# ////// SING //////
# target_sources(dino_lib 
#     PRIVATE
#     ${CMAKE_CURRENT_SOURCE_DIR}/Sing.cpp
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