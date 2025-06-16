# diNo Library

**diNoSimilaritySearch** is a modern C++ and Python library for approximate nearest neighbor search. It includes:

- C++ core library
- Python bindings
- Benchmarking tools
- GoogleTest integration

## Table of Contents

1. [Requirements](#requirements)
2. [Installation](#installation)
   - 2.1. [Installing tkinter](#installing-tkinter)
   - 2.2. [Installing MPI](#installing-mpi)
3. [Building with CMake](#building-with-cmake)
4. [Running Tests](#running-tests)
5. [Environment Setup](#environment-setup)

   - 5.1. [diNo Environment](#dino-environment)
   - 5.2. [FAISS Environment](#faiss-environment)

6. [Notes](#notes)

## Requirements

- **FAISS compatibility**: Python 3.8.16 / 3.9.17 / 3.10.13 (recommended)
- **diNo Python API**: Python 3.12
- **C++ Build**: Requires C++14 or higher
- **CMake**: Version 3.15+
- **GoogleTest**: Included via CMake
- **GoogleBenchmark**: Included via CMake
- **MPI**: Required for Odyssey algorithm

## Installation

### Submodules

Before building, initialize submodules:

- `git submodule update --init --recursive`

### Installing `tkinter`

- Linux (Ubuntu/Debian)

```bash
sudo apt install python3.12-tk
```

(Adjust version as needed.)

- macOS

```bash
brew install python-tk
```

- Windows

`tkinter` is included in the standard Python installer. Ensure "Tcl/Tk and IDLE" is selected during installation.

### Installing `MPI`

- Linux (Ubuntu/Debian)

```bash
sudo apt update
sudo apt install openmpi-bin openmpi-common libopenmpi-dev
```

- macOS

```bash
brew install open-mpi
```

- Windows

Use WSL and follow Linux steps, or install MPICH natively. Ensure consistency with your build tools.

## Building with CMake

### Basic Build (C++ only)

To build only the core C++ library (without Python bindings or anything):

```bash
mkdir build
cd build
cmake .. -DBUILD_PYTHON=OFF -DBUILD_BENCHMARK=OFF -DBUILD_DEMO=OFF
cmake --build .
```

### Optional Build Flags

- `BUILD_PYTHON=ON` – Build Python bindings (default ON)
- `BUILD_BENCHMARK=ON`– Build benchmarks (default ON)
- `BUILD_DEMO=ON` – Build demo files (default ON)
- `ODYSSEY_MPI=ON`– Enable Odyssey (default ON)
- `DEBUG_MSG=OFF`– Disable debug messages (default OFF)

### Disable Specific Components

Disable Python Bindings for example

```bash
cmake .. -DBUILD_PYTHON=OFF
cmake --build .
```

## How to run

### All C++ Files

From the `build/` folder, run:

```bash
./demos/demo_Odyssey
```

### `demo_bruteforce` && `demo_LbBruteforce`

From the `demos/` folder, run:

```bash
./demo_X.py
```

'X' is the name of the algo.

### `demo_Odyssey`

From `demos/` folder, run with MPI:

```bash
mpirun -np 4 python ../demos/demo_Odyssey.py
```

### Running Tests

Use CTest for running unit tests:

```bash
mkdir build
cd build
cmake -S . -B build
cmake --build build
ctest --test-dir build --output-on-failure
```

## Environment Setup

### diNo Environment

#### Option 1: Using pip

1. Create a virtual environment (optional but recommended):

```bash
python3.12 -m venv diNo_env
```

2. Activate the virtual environment:

- macOS/Linux:

```bash
source diNo_env/bin/activate
```

- Windows:

```bash
.\diNo_env\Scripts\activate
```

3. Install dependencies:

From your `diNo_env/`:

```bash
pip install -r requirements_diNo.txt
```

4. Deactivate the environment (when you're done):

```bash
deactivate
```

### FAISS Environment

#### Option 1: Using pip

1. Create a virtual environment (optional but recommended):

```bash
python3.10 -m venv faiss_env
```

2. Activate the virtual environment:

- macOS/Linux:

```bash
source faiss_env/bin/activate
```

- Windows:

```bash
.\faiss_env\Scripts\activate
```

3. Install dependencies:

From your `faiss_env/`:

```bash
pip install -r requirements_faiss.txt
```

4. Deactivate the environment (when you're done):

```bash
deactivate
```

#### Option 2: Using Conda

```bash
conda env create -f environment_faiss.yml
```

_Replace `my_custom_env` with your preferred environment name._

## Notes

- The compiled `.so` (shared object) file from pybind is located in the /demo folder.
- Run all Python scripts from within the `/demo` directory to ensure correct imports and paths.
