# diNo Similarity Search Library

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![C++](https://img.shields.io/badge/C%2B%2B-14%2B-blue.svg)](https://isocpp.org/)
[![Python](https://img.shields.io/badge/Python-3.10%2B-green.svg)](https://www.python.org/)
[![CMake](https://img.shields.io/badge/CMake-3.15%2B-red.svg)](https://cmake.org/)
[![Platform](https://img.shields.io/badge/Platform-Linux%20%7C%20macOS%20%7C%20Windows-lightgrey.svg)](https://github.com/MChatzakis/diNo-lib)

**diNoSimilaritySearch** is a high-performance C++ and Python library designed for efficient nearest neighbor search on time series data. The library provides multiple algorithmic implementations optimized for various computational environments and use cases, from single-machine processing to distributed computing and GPU acceleration.

## 🚀 Key Features

- **Multi-Algorithm Support**: Comprehensive suite of similarity search algorithms
- **Cross-Platform Compatibility**: Native support for Linux, macOS, and Windows
- **Language Bindings**: High-performance C++ core with intuitive Python bindings
- **Scalability**: Optimized for both single-machine and distributed computing
- **GPU Acceleration**: CUDA support for high-throughput processing
- **Comprehensive Testing**: Extensive test suite with GoogleTest framework
- **Performance Analysis**: Built-in benchmarking tools for algorithm comparison

## 🏗️ Architecture Overview

### Core Algorithms

| Algorithm | Description | Use Case |
|-----------|-------------|----------|
| **Brute Force** | Basic similarity search implementation | Baseline performance, small datasets |
| **Lower Bound Brute Force** | Optimized brute force with early termination | Medium datasets, exact results |
| **MESSI** | Memory-efficient indexing for exact DTW searches | Large in-memory datasets |
| **PARIS** | Parallel indexing for disk-based datasets | Large datasets with disk constraints |
| **Odyssey** | MPI-based distributed search algorithm | Massive datasets across multiple nodes |
| **SING** | CUDA-accelerated GPU parallel processing | High-throughput GPU environments |

## 📋 Prerequisites

### System Requirements

- **Operating System**: Linux, macOS, or Windows
- **C++ Compiler**: C++14 or higher (GCC 6+, Clang 3.4+, MSVC 2015+)
- **CMake**: Version 3.15 or higher
- **Python**: 3.10-3.12 (3.12 recommended for diNo, 3.10 for FAISS)

### Optional Dependencies

- **MPI**: Required for Odyssey distributed computing algorithm
- **CUDA**: Required for SING GPU acceleration algorithm
- **tkinter**: Required for graphical user interface demonstrations

## 🛠️ Installation

### 1. Repository Setup

```bash
# Clone the repository
git clone https://github.com/MChatzakis/diNo-lib.git
cd diNo-lib

# Initialize submodules
git submodule update --init --recursive
```

### 2. Environment Configuration

#### Python Virtual Environment (Recommended)

```bash
# Create virtual environment
python3.12 -m venv diNo_env

# Activate environment
source diNo_env/bin/activate          # Linux/macOS
# or
.\diNo_env\Scripts\activate          # Windows

# Install dependencies
pip install -r requirements_diNo.txt
```

#### Conda Environment

```bash
conda env create -f environment_diNo.yml
conda activate diNo_env
```

### 3. Build Configuration

```bash
# Create build directory
mkdir build && cd build

# Configure with CMake
cmake ..

# Build the project
cmake --build .
```

## ⚙️ Build Options

### CMake Configuration Flags

| Flag | Description | Default | Dependencies |
|------|-------------|---------|--------------|
| `BUILD_PYTHON` | Enable Python bindings | `ON` | Python 3.10+ |
| `BUILD_BENCHMARK` | Build benchmarking tools | `ON` | GoogleBenchmark |
| `BUILD_DEMO` | Build demonstration applications | `ON` | Core library |
| `ODYSSEY_MPI` | Enable MPI for distributed computing | `ON` | OpenMPI/MPICH |
| `SING_CUDA` | Enable CUDA for GPU acceleration | `ON` | CUDA Toolkit |
| `DEBUG_MSG` | Enable debug output | `OFF` | None |

### Common Build Configurations

#### Minimal Build (C++ Only)
```bash
cmake .. -DBUILD_PYTHON=OFF -DBUILD_BENCHMARK=OFF -DBUILD_DEMO=OFF
cmake --build .
```

#### CPU-Only Build
```bash
cmake .. -DODYSSEY_MPI=OFF -DSING_CUDA=OFF
cmake --build .
```

#### Build con CUDA (per testare la demo Sing)
Richiede **CUDA Toolkit** installato (`nvcc --version` e `nvidia-smi` funzionanti). La demo `demo_Sing_L2Square` viene compilata solo se CUDA è disponibile.

```bash
mkdir -p build && cd build
# Se non usi MPI (consigliato per test locali):
cmake .. -DODYSSEY_MPI=OFF -DSING_CUDA=ON -DBUILD_DEMO=ON
# Oppure con MPI:
# cmake .. -DSING_CUDA=ON -DBUILD_DEMO=ON

cmake --build . -j
./demos/demo_Sing_L2Square
```

- **Architettura GPU**: di default è `75` (Turing). Se hai un’GPU diversa imposta ad es. `-DCMAKE_CUDA_ARCHITECTURES=86` (Ampere) o `89` (Ada). Controlla [CUDA arch list](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#gpu-feature-list).
- Se la configurazione segnala "CUDA toolkit not found", verifica `PATH` e `LD_LIBRARY_PATH` (vedi `docs/cuda-installation.md`).

#### Debug Build
```bash
cmake .. -DCMAKE_BUILD_TYPE=Debug -DDEBUG_MSG=ON
cmake --build .
```

## 🚀 Quick Start

### Basic Usage Example

```python
import numpy as np
from diNo import BruteForceL2Square

# Create sample data
data = np.random.randn(1000, 64)
query = np.random.randn(64)

# Initialize search algorithm
searcher = BruteForceL2Square(data)

# Perform search
distances, indices = searcher.search(query, k=5)
print(f"Top 5 matches: {indices}")
print(f"Distances: {distances}")
```

### C++ Example

```cpp
#include <diNo/bruteforce_l2square.hpp>
#include <vector>

int main() {
    std::vector<std::vector<float>> data = {{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}};
    std::vector<float> query = {1.0, 2.0, 3.0};
    
    auto searcher = diNo::BruteForceL2Square(data);
    auto results = searcher.search(query, 2);
    
    return 0;
}
```

## 📚 Usage Examples

### Running Demonstrations

#### C++ Demos
```bash
cd build

# Basic algorithms
./demos/demo_bruteforce_L2Square
./demos/demo_bruteforce_DTW
./demos/demo_LbBruteforce_L2Square

# Advanced algorithms
./demos/demo_Messi_L2Squares
./demos/demo_Paris_L2Square
./demos/demo_Odyssey_L2Square    # MPI required
./demos/demo_Sing_L2Square       # CUDA required
```

#### Python Demos
```bash
cd demos

# Basic demonstrations
python3.12 demo_bruteforce_L2Square.py
python3.12 demo_Messi_L2Square.py

# Interactive GUI
python3.12 demo_gui.py

# Distributed computing (MPI)
mpirun -np 4 python3.12 demo_Odyssey_L2Square.py

# GPU acceleration (CUDA)
python3.12 demo_Sing_L2Square.py
```

## 🧪 Testing

### Running Test Suite

```bash
cd build

# Complete test suite
ctest --output-on-failure

# Individual test execution
./tests/test_bruteforce_L2Square
./tests/test_Messi_L2Square
./tests/test_Odyssey_L2Square    # MPI required
./tests/test_Sing_L2Square       # CUDA required

# Verbose output
ctest --output-on-failure --verbose
```

## 📊 Performance Analysis

### Benchmarking

Execute performance benchmarks to compare algorithm performance:

```bash
cd build

# Core algorithms
./benchmark/bm_bruteforce_L2Square
./benchmark/bm_LbBruteforce_L2Square
./benchmark/bm_Messi_L2Square

# Advanced algorithms (if available)
./benchmark/bm_Odyssey_L2Square    # MPI required
./benchmark/bm_Sing_L2Square       # CUDA required
```

## 🔧 Troubleshooting

### Common Issues and Solutions

#### Build Errors
- **Submodule Issues**: Ensure `git submodule update --init --recursive` is executed
- **CMake Version**: Verify CMake 3.15+ is installed (`cmake --version`)
- **Compiler Support**: Confirm C++14+ compatibility

#### Python Import Errors
- **Path Issues**: Execute Python scripts from the `demos/` directory
- **Environment**: Ensure virtual environment is activated
- **Bindings**: Verify Python bindings were built (`BUILD_PYTHON=ON`)

#### MPI Configuration
- **Installation**: Verify MPI installation (`mpirun --version`)
- **Process Count**: Ensure MPI process count matches available cores
- **Implementation**: Use consistent MPI implementation across build and runtime

#### CUDA Issues
- **Toolkit**: Verify CUDA installation (`nvcc --version`)
- **Hardware**: Check GPU compatibility and driver version (`nvidia-smi`)
- **Environment**: Ensure CUDA toolkit path is properly configured

## 📖 Documentation

For detailed documentation and advanced usage examples, please refer to:

- **API Reference**: Comprehensive documentation of all classes and methods
- **Algorithm Details**: In-depth explanation of each search algorithm
- **Performance Guide**: Optimization strategies and best practices
- **Contributing**: Guidelines for contributing to the project

## 🤝 Contributing

We welcome contributions from the community! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details on:

- Code style and standards
- Testing requirements
- Pull request process
- Development setup

## 📄 License

This project is licensed under the [MIT License](LICENSE) - see the [LICENSE](LICENSE) file for complete details.

## 🙏 Acknowledgments

Special thanks to the open-source community and contributors who have helped make this library possible.

---

**diNoSimilaritySearch** - Efficient Time Series Similarity Search for Modern Computing Environments
