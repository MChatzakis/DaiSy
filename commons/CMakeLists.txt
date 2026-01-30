if(DEBUG_MSG)
    message(STATUS "--- Configuring commons_lib ---")
endif()

# Create a commons library
if(DEBUG_MSG)
    message(STATUS "Attempting to add library: commons_lib")
endif()
add_library(commons_lib
    dataloaders.cpp
    test_bm_utils.cpp
    paramSetup.cpp
    common.cpp
)
if(DEBUG_MSG)
    message(STATUS "Library commons_lib added.")
endif()

# Required when this library is linked into a shared object (e.g. Python extension)
set_target_properties(commons_lib PROPERTIES POSITION_INDEPENDENT_CODE ON)

# Make its headers visible to whoever links it
if(DEBUG_MSG)
    message(STATUS "Setting public include directories for commons_lib...")
endif()
target_include_directories(commons_lib PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)
if(DEBUG_MSG)
    message(STATUS "Public include directories set for commons_lib.")
endif()

# Optionally set flags for this library
#target_compile_options(commons_lib PRIVATE -Wall -Wextra -O2)

if(DEBUG_MSG)
    message(STATUS "--- commons_lib Configuration Complete ---")
endif()