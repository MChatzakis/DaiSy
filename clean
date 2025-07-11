#!/bin/bash

BUILD_DIR="build" 

echo "Starting clean process..."

# Check if the build directory exists
if [ -d "$BUILD_DIR" ]; then
    echo "Removing build directory: $BUILD_DIR"
    rm -rf "$BUILD_DIR"
    echo "Successfully removed $BUILD_DIR"
    mkdir build/
    echo "Successfully created $BUILD_DIR"
else
    echo "Build directory '$BUILD_DIR' not found. Nothing to remove."
fi

echo "Clean process finished."