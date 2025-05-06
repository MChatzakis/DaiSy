# Create a commons library
add_library(commons_lib
    dataloaders.cpp
)

# Make its headers visible to whoever links it
target_include_directories(commons_lib PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

# Optionally set flags for this library
#target_compile_options(commons_lib PRIVATE -Wall -Wextra -O2)