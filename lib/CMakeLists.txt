# Create a library target from source files
add_library(
    dino_lib
    distance_computers/DistanceComputer.cpp
)

# Make it position-independent for use in a shared object
set_target_properties(dino_lib PROPERTIES POSITION_INDEPENDENT_CODE ON)

add_subdirectory(algos)

# Include all headers (including subfolders)
target_include_directories(
    dino_lib PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

# Optionally, if you had source files in algos/, utils/, etc — add them too:
# target_sources(dino_lib PRIVATE algos/MyAlgo.cpp utils/Utils.cpp ...)