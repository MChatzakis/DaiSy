# diNo Library

**diNoSimilaritySearch** is a high-performance C++ and Python library for nearest neighbor search on time series data. It provides multiple algorithms optimized for different use cases and computational environments.

## Features

- **Multiple Algorithms**:
  - **_Brute Force_** - Basic similarity search
  - **_Lower Bound Brute Force_** - Optimized brute force
  - **_MESSI_** - An advanced indexing algorithm that maximizes memory efficiency to enable fast, exact Dynamic Time Warping (DTW) searches on large in-memory datasets.
  - **_Odyssey_** - An MPI-based distributed search algorithm that focuses on distributing the search workload across multiple machines to efficiently handle truly massive datasets. It aims to exploit parallelism both within and across system nodes.
  - **_PARIS_** - An indexing algorithm that leverages parallelism to offer fast search times for disk-based datasets.
  - **_SING_** - A CUDA-accelerated search algorithm that leverages GPUs for parallel processing to perform exact DTW similarity searches on vast time series datasets, overcoming computational bottlenecks.
- **Cross-Platform**: Support for Linux, macOS, and Windows
- **Language Bindings**: Native C++ library with Python bindings
- **Optimized**: CUDA and MPI support for GPU and distributed computing
- **Well-Tested**: Comprehensive test suite with GoogleTest
- **Benchmarking**: Built-in performance benchmarking tools

## Quick Start

```bash
# Clone the repository
git clone https://github.com/MChatzakis/diNo-lib.git
cd diNo-lib
git submodule update --init --recursive

# Setup environment and build
python3.12 -m venv diNo_env
source diNo_env/bin/activate  # On Windows: .\diNo_env\Scripts\activate
pip install -r requirements_diNo.txt

# Build the library
mkdir -p build && cd build
cmake ..
cmake --build .

# Run a demo
./demos/demo_bruteforce_L2Square
```

## Table of Contents

1. [Features](#features)
2. [Quick Start](#quick-start)
3. [Requirements](#requirements)
4. [Installation](#installation)
   - 4.1. [Submodules](#submodules)
   - 4.2. [Dependencies](#dependencies)
5. [Environment Setup](#environment-setup)
   - 5.1. [diNo Environment](#dino-environment)
   - 5.2. [FAISS Environment](#faiss-environment)
6. [Building with CMake](#building-with-cmake)
7. [Usage](#usage)
   - 7.1. [C++ Demos](#c-demos)
   - 7.2. [Python Demos](#python-demos)
8. [Running Tests](#running-tests)
9. [Running Benchmarks](#running-benchmarks)
10. [Troubleshooting](#troubleshooting)
11. [License](#license)

## Requirements

### Core Requirements

- **C++ Compiler**: C++14 or higher
- **CMake**: Version 3.15+
- **Python**: 3.10-3.12 (3.12 recommended for diNo, 3.10 for FAISS)

### Optional Dependencies

- **MPI**: Required for Odyssey algorithm (distributed computing)
- **CUDA**: Required for SING algorithm (GPU acceleration)
- **tkinter**: Required for GUI demos

### Included via CMake

- **GoogleTest**: Unit testing framework
- **GoogleBenchmark**: Performance benchmarking

## Installation

### Submodules

Initialize git submodules before building:

```bash
git submodule update --init --recursive
```

### Dependencies

Choose the components you need:

#### Basic Setup (Core library only)

No additional dependencies required.

#### GUI Demos (tkinter)

- **Linux (Ubuntu/Debian)**:

```bash
sudo apt install python3-tk
```

- **macOS**:

```bash
brew install python-tk
```

- **Windows**: Included with Python installer (ensure "Tcl/Tk and IDLE" is selected).

#### Distributed Computing (MPI for Odyssey)

- **Linux (Ubuntu/Debian)**:

```bash
sudo apt update
sudo apt install openmpi-bin openmpi-common libopenmpi-dev
```

- **macOS**:

```bash
brew install open-mpi
```

- **Windows**: Use WSL with Linux instructions, or install MPICH natively.

#### GPU Acceleration (CUDA for SING)

- **Linux**: See [detailed CUDA installation guide](docs/cuda-installation.md)

- **macOS**: Not supported (NVIDIA discontinued macOS CUDA support)

- **Windows**: Download from [NVIDIA CUDA Downloads](https://developer.nvidia.com/cuda-downloads)

## Environment Setup

### diNo Environment

**Recommended for most users** - supports the latest diNo features.

#### Using pip (Recommended)

```bash
# Create and activate virtual environment
python3.12 -m venv diNo_env
source diNo_env/bin/activate  # Windows: .\diNo_env\Scripts\activate

# Install dependencies
pip install -r requirements_diNo.txt
```

#### Using Conda

```bash
conda env create -f environment_diNo.yml
conda activate diNo_env
```

### FAISS Environment

**For FAISS comparison benchmarks** - requires specific Python versions.

#### Using pip

```bash
# Create and activate virtual environment
python3.10 -m venv faiss_env
source faiss_env/bin/activate  # Windows: .\faiss_env\Scripts\activate

# Install dependencies
pip install -r requirements_faiss.txt
```

#### Using Conda

```bash
conda env create -f environment_faiss.yml
conda activate faiss_env
```

## Building with CMake

### Standard Build

Build everything with default options:

```bash
mkdir -p build && cd build
cmake ..
cmake --build .
```

### Build Options

Configure build components using CMake flags:

| Flag              | Description                      | Default |
| ----------------- | -------------------------------- | ------- |
| `BUILD_PYTHON`    | Build Python bindings            | ON      |
| `BUILD_BENCHMARK` | Build benchmarking tools         | ON      |
| `BUILD_DEMO`      | Build demo applications          | ON      |
| `ODYSSEY_MPI`     | Enable MPI for Odyssey algorithm | ON      |
| `SING_CUDA`       | Enable CUDA for SING algorithm   | ON      |
| `DEBUG_MSG`       | Enable debug messages            | OFF     |

### Common Build Configurations

**C++ only (minimal build)**:

```bash
cmake .. -DBUILD_PYTHON=OFF -DBUILD_BENCHMARK=OFF -DBUILD_DEMO=OFF
cmake --build .
```

**Without GPU/MPI support**:

```bash
cmake .. -DODYSSEY_MPI=OFF -DSING_CUDA=OFF
cmake --build .
```

**Debug build**:

```bash
cmake .. -DCMAKE_BUILD_TYPE=Debug -DDEBUG_MSG=ON
cmake --build .
```

## Usage

### C++ Demos

Run compiled C++ demos from the `build/` directory:

```bash
# Basic similarity search
./demos/demo_bruteforce_L2Square
./demos/demo_bruteforce_DTW

# Lower bound optimization
./demos/demo_LbBruteforce_L2Square
./demos/demo_LbBruteforce_DTW

# Advanced algorithms
./demos/demo_Messi_L2Squares
./demos/demo_Paris_L2Square
./demos/demo_Odyssey_L2Square  # Requires MPI
./demos/demo_Sing_L2Square     # Requires CUDA
```

### Python Demos

**Important**: Run Python demos from the `demos/` directory for correct imports.

```bash
cd demos
```

#### Basic Demos

```bash
python3.12 demo_bruteforce_L2Square.py
python3.12 demo_bruteforce_DTW.py
python3.12 demo_LbBruteforce_L2Square.py
python3.12 demo_LbBruteforce_DTW.py
python3.12 demo_Messi_L2Square.py
```

#### GUI Demo

```bash
python3.12 demo_gui.py
```

#### Advanced Demos

**MPI-based (Odyssey)**:

```bash
mpirun -np 4 python3.12 demo_Odyssey_L2Square.py
```

**CUDA-based (SING)**:

```bash
python3.12 demo_Sing_L2Square.py
```

## Running Tests

### All Tests

Run the complete test suite:

```bash
cd build
ctest --output-on-failure
```

### Specific Tests

Run individual test executables:

```bash
cd build
./tests/test_bruteforce_L2Square
./tests/test_bruteforce_DTW
./tests/test_LbBruteforce_L2Square
./tests/test_LbBruteforce_DTW
./tests/test_Messi_L2Square
./tests/test_Odyssey_L2Square       # Requires MPI
./tests/test_Sing_L2Square          # Requires CUDA
```

### Test with Verbose Output

```bash
ctest --output-on-failure --verbose
```

## Running Benchmarks

Execute performance benchmarks from the `build/` directory (after building with the `cmake`):

```bash
cd build

# Basic benchmarks
./benchmark/bm_bruteforce_L2Square
./benchmark/bm_LbBruteforce_L2Square
./benchmark/bm_Messi_L2Square

# Advanced benchmarks (if available)
./benchmark/bm_Odyssey_L2Square      # Requires MPI
./benchmark/bm_Sing_L2Square         # Requires CUDA
```

## Troubleshooting

### Common Issues

**Build Errors**:

- Ensure all submodules are initialized: `git submodule update --init --recursive`
- Check CMake version: `cmake --version` (requires 3.15+)
- Verify C++ compiler supports C++14 or higher

**Python Import Errors**:

- Run Python scripts from the `demos/` directory
- Ensure virtual environment is activated
- Check that Python bindings were built (`BUILD_PYTHON=ON`)

**MPI Issues**:

- Verify MPI installation: `mpirun --version`
- Check MPI process count matches your system cores
- Ensure consistent MPI implementation across build and runtime

**CUDA Issues**:

- Verify CUDA installation: `nvcc --version`
- Check GPU compatibility and driver version: `nvidia-smi`
- Ensure CUDA toolkit path is in environment variables

## License

This project is licensed under the [MIT License](LICENSE) - see the LICENSE file for details.
