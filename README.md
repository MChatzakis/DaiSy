# diNo Library

**diNoSimilaritySearch** is a library for approximate nearest neighbor search with support for modern C++ and Python bindings. It includes benchmarking tools, GoogleTest integration, and support for comparison with FAISS.

## Requirements

- **FAISS compatibility**: Python 3.8.16 / 3.9.17 / 3.10.13 (recommended)
- **diNo Python API**: Python 3.12
- **C++ Build**: Requires a compiler with C++14 support (or higher)
- **CMake**: Version 3.15 or higher recommended
- **GoogleTest**: Integrated for unit testing (included via CMake)
- **GoogleBenchmark**:
- **Note**: This project uses `tkinter`, which is part of the Python standard library but may require separate installation.

## Install

First, all submodules should be updated:

- `git submodule update --init --recursive`

Then, the instructions for the operating system should be followed:

### Installing `tkinter`

`tkinter` is part of the Python standard library, but its Tcl/Tk backend might need to be installed separately, especially on Linux.

- Linux (Ubuntu/Debian)

```bash
sudo apt install python3.12-tk
```

(Adjust python3.12-tk to match your Python version if it's different, e.g., python3.14-tk).

- macOS

If Python was installed via _Homebrew_, `tkinter` can be ensured to be correctly linked by running:

```bash
brew install python-tk
```

If the official Python installer from python.org was used, `tkinter` is usually included by default. One should just make sure the "Tcl/Tk and IDLE" option was selected during installation.

- Windows

When installing Python from the official python.org website, tkinter is typically included by default. Confirm that the "Tcl/Tk and IDLE" option is selected during the installation process.

## Building with CMake

### Basic Build (C++ library only)

To build only the core C++ library (without Python bindings or benchmarks):

```bash
mkdir build
cd build
cmake .. -DBUILD_PYTHON=OFF -DBUILD_BENCHMARK=OFF -DBUILD_DEMO=OFF
cmake --build .
```

Note: `BUILD_PYTHON`, `BUILD_BENCHMARK`, and `BUILD_DEMO` are optional flags:

- `BUILD_PYTHON=ON` (default): Enables building Python bindings.
- `BUILD_BENCHMARK=ON` (default): Enables building benchmarks.
- `BUILD_DEMO=ON` (defaul): Enables building demonstration files.

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

### Disables Demos

```bash
cmake .. -DBUILD_DEMO=OFF
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

## Installation for diNo library

### Option 1: Using pip

#### 1. Create a virtual environment (optional but recommended):

```bash
python3.12 -m venv diNo_env
```

#### 2. Activate the virtual environment:

- macOS/Linux:

```bash
source diNo_env/bin/activate
```

- Windows:

```bash
.\diNo_env\Scripts\activate
```

#### 3. Install dependencies:

From your `diNo_env/`:

```bash
pip install -r requirements_diNo.txt
```

#### 4. Deactivate the environment (when you're done):

```bash
deactivate
```

## Installation for FAISS

### Option 1: Using pip

#### 1. Create a virtual environment (optional but recommended):

```bash
python3.10 -m venv faiss_env
```

#### 2. Activate the virtual environment:

- macOS/Linux:

```bash
source faiss_env/bin/activate
```

- Windows:

```bash
.\faiss_env\Scripts\activate
```

#### 3. Install dependencies:

From your `faiss_env/`:

```bash
pip install -r requirements_faiss.txt
```

#### 4. Deactivate the environment (when you're done):

```bash
deactivate
```

### Option 2: Using Conda

Note: change the `my_custom_env` in .yml to your env name.

```bash
conda env create -f environment_faiss.yml
```
