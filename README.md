# diNo Library

**diNoSimilaritySearch** is a library for approximate nearest neighbor search with support for modern C++ and Python bindings. It includes benchmarking tools, GoogleTest integration, and support for comparison with FAISS.

## Requirements

- **FAISS compatibility**: Python 3.8.16 / 3.9.17 / 3.10.13 (recommended)
- **diNo Python API**: Python 3.12
- **C++ Build**: Requires a compiler with C++14 support (or higher)
- **CMake**: Version 3.15 or higher recommended
- **GoogleTest**: Integrated for unit testing (included via CMake)
- **GoogleBenchmark**:

## Install

- `git submodule update --init --recursive`

## Building with CMake

### Basic Build (C++ library only)

To build only the core C++ library (without Python bindings or benchmarks):

```bash
mkdir build
cd build
cmake .. -DBUILD_PYTHON=OFF -DBUILD_BENCHMARK=OFF
cmake --build .
```

Note: `BUILD_PYTHON` and `BUILD_BENCHMARK` are both optional flags:

- `BUILD_PYTHON=ON` (default): Enables building Python bindings.

- `BUILD_BENCHMARK=ON` (default): Enables building benchmarks.

### Disables Python Bindings

```bash
cmake .. -DBUILD_PYTHON=OFF
cmake --build .
```

### Disables Benchmarks

```bash
cmake .. -DBUILD_BENCHMARK=OFF
cmake --build .
```

## Running Tests with CTest

```bash
mkdir build
cd build
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
