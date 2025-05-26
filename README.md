# diNo Library

diNoSimilaritySearch is a library for approximate nearest neighbor search with support for modern C++ and Python bindings. It includes benchmarking tools, GoogleTest integration, and support for comparison with FAISS.

## Requirements

- FAISS compatibility: Python 3.8.16 / 3.9.17 / 3.10.13 (recommended)
- diNo Python API: Python 3.12
- C++ Build: Requires a compiler with C++14 support (or higher)
- CMake: Version 3.15 or higher is recommended
- GoogleTest: Integrated for unit testing (included via CMake)

## Building with CMake

Note: To build only the C++ version of the library, use `-DBUILD_PYTHON=OFF`. Otherwise, you can omit the flag.

```bash
mkdir build
cd build
cmake .. -DBUILD_PYTHON=OFF  # ON by default
cmake --build .
```

## Running Tests with CTest

```bash
cmake -S . -B build
cmake --build build
ctest --test-dir build --output-on-failure
```

## Installation for FAISS

### Option 1: Using pip

From your `env/`:

```bash
pip install -r requirements.txt
```

### Option 2: Using Conda

Note: change the `my_custom_env` in .yml to your env name.

```bash
conda env create -f environment.yml
```
