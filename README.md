# DaiSy: A Library for Scalable Data Series Similarity Search

DaiSy is a unified library for exact data series similarity search that integrates multiple state-of-the-art algorithms within a single, coherent framework, developed at LIPADE, Université Paris Cit\'e.
It supports a wide range approaches tailored for different execution environments, including disk-based, in-memory, GPU-accelerated, and distributed scalable similarity search. 
DaiSy is implemented in C++, while it also offers a convenient Python interface for ease of use and integration with data science workflows.


## Supported State-of-the-Art algorithms

We currently support several algorithms for exact similarity search, each optimized for specific use cases and environments. 
The following table summarizes the key features of each algorithm:

| Algorithm | Description |
|-----------|-------------|
| **Bruteforce** | Naive sequential similarity search implementation |
| **Lower Bound Bruteforce** | Optimized brute force with lower bounding for the distance calculations |
| **MESSI** | In-memory parallel similarity search |
| **PARIS** | Disk-based parallel similarity search |
| **SING** | GPU-accelerated in-memory parallel similarity search |
| **Odyssey** | Distributed and parallel in-memory similarity search |



## Quickstart

### Dependencies
- **Operating System**: Linux, macOS, or Windows
- **C++ Compiler**: C++14 or higher (GCC 6+, Clang 3.4+, MSVC 2015+)
- **CMake**: Version 3.15 or higher

Optionally,

- **Python**: 3.10-3.12
- **MPI**: Required for Odyssey distributed computing algorithm
- **CUDA**: Required for SING GPU acceleration algorithm


### Installation
To download DaiSy, use:
```bash
git clone https://github.com/MChatzakis/daisy.git
cd daisy
git submodule update --init --recursive
```

Based on the available hardware, you can specify the below arguments to enable/disable features.
| Flag | Description | Default | Dependencies |
|------|-------------|---------|--------------|
| `BUILD_PYTHON` | Enable Python bindings | `ON` | Python 3.10+ |
| `BUILD_BENCHMARK` | Build benchmarking tools | `ON` | GoogleBenchmark |
| `BUILD_DEMO` | Build demonstration applications | `ON` | Core library |
| `ODYSSEY_MPI` | Enable MPI for distributed computing | `ON` | OpenMPI/MPICH |
| `SING_CUDA` | Enable CUDA for GPU acceleration | `ON` | CUDA Toolkit |
| `DEBUG_MSG` | Enable debug output | `OFF` | None |

To compile:
```bash
# Create build directory
mkdir build && cd build

# Configure with CMake
cmake ..

# Build the project
cmake --build .
```

### Enable Python

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

### Running the test suite


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

## To run a performance analysis
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

## License
DaiSy licensed under the [MIT License](LICENSE) - see the [LICENSE](LICENSE) file for complete details.

## About
DaiSy is developed by the diNo research group at LIPADE, Université Paris Cit\'e. It is provided with no warranty, and we encourage contributions from the community to enhance its capabilities and performance. For questions, issues, or contributions, please open an issue or submit a pull request on GitHub.










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
`



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


