# Knob-optimizer tools. Built only with -DBUILD_OPTIMIZER=ON.
#
# optimizer_collector: runs MESSI, FreSh, SOFA and Odyssey (single node) over a
# grid of knob values and writes one CSV row (knobs -> latency) per query.
# SOFA and Odyssey are included only when their optional dependencies were found.

find_package(Threads REQUIRED)

add_executable(optimizer_collector optimizer_collector.cpp)
target_link_libraries(optimizer_collector PRIVATE dino_lib commons_lib Threads::Threads)

if(BUILD_ODYSSEY_AVAILABLE)
    target_link_libraries(optimizer_collector PRIVATE MPI::MPI_CXX)
endif()
