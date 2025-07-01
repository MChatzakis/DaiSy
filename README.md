# diNo Library

**diNoSimilaritySearch** is a C++ and Python library for approximate nearest neighbor search. It includes:

- C++ core library
- Python bindings
- Benchmarking tools
- GoogleTest integration

## Table of Contents

1. [Requirements](#requirements)
2. [Installation](#installation)
   - 2.1. [Submodules](#submodules)
   - 2.2. [Installing tkinter](#installing-tkinter)
   - 2.3. [Installing MPI](#installing-mpi)
   - 2.4. [Installing CUDA](#installing-cuda)
3. [Environment Setup](#environment-setup)
   - 3.1. [diNo Environment](#dino-environment)
   - 3.2. [FAISS Environment](#faiss-environment)
4. [Building with CMake](#building-with-cmake)
5. [How to run Demos](#how-to-run-demos)
   - 5.1. [C++ Demos](#c-demos)
   - 5.2. [Python Demos](#python-demos)
6. [Running Tests](#running-tests)
7. [Running Benchmarks](#running-benchmarks)
8. [Notes](#notes)

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

- **Linux** (Ubuntu/Debian)

  ```bash
  sudo apt install python3.12-tk
  ```

(Adjust version as needed.)

- **macOS**

  ```bash
  brew install python-tk
  ```

- **Windows**

  `tkinter` is included in the standard Python installer. Ensure "Tcl/Tk and IDLE" is selected during installation.

### Installing `MPI` for Odyssey

- **Linux** (Ubuntu/Debian)

  ```bash
  sudo apt update
  sudo apt install openmpi-bin openmpi-common libopenmpi-dev
  ```

- **macOS**

  ```bash
  brew install open-mpi
  ```

- **Windows**

  Use WSL and follow Linux steps, or install MPICH natively. Ensure consistency with your build tools.

### Installing `CUDA` for SING

- **Linux** (Ubuntu)

  1. Install Required Packages

  ```bash
  sudo apt update
  sudo apt install curl
  ```

  _Updates package lists and installs curl for downloading files from the internet._

  2. Install NVIDIA Drivers

  ```bash
  sudo ubuntu-drivers autoinstall
  ```

  _Note: While `ubuntu-drivers` autoinstall is convenient, for specific driver versions or more control (often preferred for CUDA), you might use `sudo apt install nvidia-driver-XXX` (e.g., `nvidia-driver-535`) after checking available drivers with `ubuntu-drivers devices`._

  Then, make sure to reboot your system after driver installation.

  ```bash
  sudo reboot
  ```

  3. Add the NVIDIA GPG Key

  ```bash
  curl -fsSL https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/3bf863cc.pub | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-cuda-keyring.gpg
  ```

  The curl `-fsSL` command ensures a silent, secure, and error-handled download.
  The downloaded key is then piped (`|`) to `sudo gpg --dearmor`, which converts it into a binary format suitable for APT and saves it as `nvidia-cuda-keyring.gpg` in the `/usr/share/keyrings/` directory.

  4. Add the CUDA Repository

  ```bash
  echo "deb [signed-by=/usr/share/keyrings/nvidia-cuda-keyring.gpg] https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/ /" | sudo tee /etc/apt/sources.list.d/cuda-repository.list
  ```

  _Adds the NVIDIA CUDA repository to your system's software sources._

  5. Update the Package Repository

  ```bash
  sudo apt update
  ```

  6. Install the CUDA Toolkit

  ```bash
  sudo apt install cuda
  ```

  7. Set Up Environment Variables (Persistent)

  ```bash
  echo 'export PATH=/usr/local/cuda/bin${PATH:+:${PATH}}' >> ~/.bashrc
  echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}' >> ~/.bashrc
  ```

  _To make environment variables persistent, you should append them to a shell configuration file like ~/.bashrc (for Bash users). Using export directly in the terminal only sets them for the current session._

  ```bash
  source ~/.bashrc
  ```

  _Applies the updated environment variables to your current shell session._

  8. Verification

  ```bash
  nvcc --version
  ```

- **macOS**

  ```bash

  ```

- **Windows**

  1.  Check System Compatibility

      - NVIDIA GPU: _Ensure your system has an NVIDIA GPU. You can check this in Device Manager under "Display adapters."_
      - Windows Version: _Ensure your Windows version is supported by the CUDA Toolkit you intend to install._

  2.  Update NVIDIA Drivers

      - It's highly recommended to have the latest NVIDIA display drivers installed.
      - Download them from the official NVIDIA driver download page: https://www.nvidia.com/drivers
      - Select your product type, series, product, operating system, and language.
      - Download and run the installer, choosing "Express Installation."

  3.  Download CUDA Toolkit

      - Go to the NVIDIA CUDA Toolkit Download page: https://developer.nvidia.com/cuda-downloads
      - Select your operating system (Windows), architecture (x86_64), and Windows version.
      - Choose the installer type. The `exe (local)` installer downloads everything, while `exe (network)` downloads components during installation. For most users, the `exe (local)` is recommended.
      - Download the installer executable.

  4.  Install CUDA Toolkit

      - Run the downloaded `.exe` installer as an administrator (right-click -> "Run as administrator").
      - Temporary Extract Location: The installer will first ask for a temporary extraction path. You can keep the default or choose a different location.
      - System Check: _The installer will perform a system check._
      - License Agreement: _Read and accept the NVIDIA Software License Agreement._
      - Installation Options:
        - Choose "Express" installation for typical setups. This installs all components in default locations.
        - Choose "Custom (Advanced)" if you want to select specific components (e.g., only the Toolkit, not the Visual Studio integration if you don't use VS) or change installation paths. For most users, "Express" is sufficient.
      - Installation Process: _The installer will proceed to install the CUDA Toolkit, including necessary drivers (if not already updated), development tools, and libraries._
      - Finish: _Once the installation is complete, click "Next" and then "Finish." You may be prompted to restart your computer._

  5.  Verify Installation
      - Environment Variables: _The CUDA installer typically sets the necessary environment variables automatically (e.g., `CUDA_PATH`, `PATH` to include `CUDA_PATH\bin` and `CUDA_PATH\lib\x64`). You can verify this by searching for "Environment Variables" in the Windows search bar and checking "System variables."_
      - Command Prompt Verification:
        - Open Command Prompt (search for `cmd` and open).
        - Run the following command to check the `nvcc` compiler version:
        ```bash
        nvcc --version
        ```
        - You should see output indicating the CUDA Toolkit version.
        - You can also run the NVIDIA System Management Interface:
        ```bash
        nvidia-smi
        ```
        - This will display information about your NVIDIA GPU, driver version, and CUDA version supported by the driver.

## Environment Setup

### diNo Environment

#### Option 1: Using pip

1. Create a virtual environment (optional but recommended):

   ```bash
   python3.12 -m venv diNo_env
   ```

_Note: replace `python3.12` with your current version._

2. Activate the virtual environment:

- **macOS/Linux**:

  ```bash
  source diNo_env/bin/activate
  ```

- **Windows**:

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

#### Option 2: Using Conda

```bash
conda env create -f environment_diNo.yml
```

_Replace `my_custom_env` with your preferred environment name._

### FAISS Environment

#### Option 1: Using pip

1. Create a virtual environment (optional but recommended):

   ```bash
   python3.10 -m venv faiss_env
   ```

2. Activate the virtual environment:

   - **macOS/Linux**:

   ```bash
   source faiss_env/bin/activate
   ```

   - **Windows**:

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

- `DEBUG_MSG=OFF`– Disable debug messages (default OFF)
- `BUILD_PYTHON=ON` – Build Python bindings (default ON)
- `BUILD_BENCHMARK=ON`– Build benchmarks (default ON)
- `BUILD_DEMO=ON` – Build demo files (default ON)
- `ODYSSEY_MPI=ON`– Enable Odyssey (default ON)
- `SING_CUDA=ON`– Enable Sing (default ON)
- _Note: If you turn off a flag, you need to explicitly turn it back on if you want to enable that component later._

### Disable Specific Components

Disable Python Bindings for example

```bash
cmake .. -DBUILD_PYTHON=OFF
cmake --build .
```

### Disable everything

```bash
cmake .. -DBUILD_PYTHON=OFF -DBUILD_BENCHMARK=OFF -DBUILD_DEMO=OFF -DODYSSEY_MPI=OFF -DSING_CUDA=OFF
cmake --build .
```

## How to run Demos

### C++ Demos

To run any compiled C++ demo, navigate to your `build/` directory and execute the specific demo file. For example, to run `demo_Odyssey`:

```bash
./demos/demo_Odyssey
```

### Python Demos

The diNo library includes several Python-based demos. Ensure you run these scripts from within the `demos/` directory itself for correct imports and paths.

#### `demo_bruteforce` and `demo_LbBruteforce`

For these demos, run these commands:

```bash
./demo_X.py
```

_'X' is the name of the algo (e.g., `./demo_bruteforce`)._

#### `demo_Odyssey`

To run the demo_Odyssey Python script, which utilizes MPI, execute it from the demos/ folder using mpirun:

```bash
mpirun -np 4 python ../demos/demo_Odyssey.py
```

## Running Tests

### Running All Tests

Use CTest for running all the unit tests:

```bash
mkdir build # Note: If it already exists, this command will give an error unless you add '-p' to make it 'mkdir -p build'.
cd build    # Note: This directory must exist for the command to succeed.
cmake -S . -B build
cmake --build build
ctest --test-dir build --output-on-failure
```

### Running Specific Tests

After building your project with CMake, navigate to your `build/` directory, and then run it directly:

```bash
./tests/test_X_Y
```

_'X' is the name of the algo with 'Y' name of the distanceComputer (e.g., `./tests/test_bruteforce_L2Square`)._

## Running Benchmarks

After building with CMake, navigate to the build directory and run your benchmark executable directly:

```bash
./benchmark/bm_X_Y
```

_'X' is the name of the algo with 'Y' name of the distanceComputer (e.g., `./benchmarks/bm_bruteforce_L2Square`)._

## Notes

- The compiled `.so` (shared object) file from pybind is located in the /demo folder.
- Run all Python scripts from within the `/demo` directory to ensure correct imports and paths.
