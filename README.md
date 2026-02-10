# DaiSy: A Library for Scalable Data Series Similarity Search
DaiSy (DAta series sImilarity sSearch librarY) is a unified library for exact data series similarity search that integrates multiple state-of-the-art algorithms within a single, coherent framework, developed at LIPADE, Université Paris Cit\'e.
It supports a wide range approaches tailored for different execution environments, including disk-based, in-memory, GPU-accelerated, and distributed scalable similarity search. 
DaiSy is implemented in C++, while it also offers a convenient Python interface for ease of use and integration with data science workflows.

<!--
When using DaiSy, please consider citing the following paper:

```latex
@inproceedings{todo,
  title={DaiSy: A Library for Scalable Data Series Similarity Search},
  author={},
  conference={Coming Soon!},
  year={2026}
}
```
-->


## Supported State-of-the-Art algorithms
We currently support several algorithms for exact similarity search, each optimized for specific use cases and environments. 
The following table summarizes the key features of each algorithm:

| Algorithm | Description |
|-----------|-------------|
| **Bruteforce** | Naive parallel similarity search implementation |
| **Lower Bound Bruteforce** | Optimized bruteforce with lower bounding for the distance calculations |
| **[MESSI](https://helios2.mi.parisdescartes.fr/~themisp/messi/)** | In-memory parallel similarity search |
| **[PARIS](https://helios2.mi.parisdescartes.fr/~themisp/paris/)** | Disk-based parallel similarity search |
| **[SING](https://helios2.mi.parisdescartes.fr/~themisp/sing/)** | GPU-accelerated in-memory parallel similarity search |
| **[Odyssey](https://helios2.mi.parisdescartes.fr/~themisp/odyssey/)** | Distributed and parallel in-memory similarity search |


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
| `BUILD_TESTS` | Build test suite | `ON` | GoogleTest |
| `BUILD_DEMO` | Build demonstration applications | `ON` | Core library |
| `ODYSSEY_MPI` | Enable MPI for distributed computing | `ON` | OpenMPI/MPICH |
| `SING_CUDA` | Enable CUDA for GPU acceleration | `ON` | CUDA Toolkit |
| `DEBUG_MSG` | Enable debug output | `OFF` | None |

To compile:
```bash
mkdir build && cd build

cmake ..
make
```

### Enable Python

```bash
python3.12 -m venv DaiSy_env

source DaiSy_env/bin/activate 
pip install -r requirements_DaiSy.txt
```


### Example Usage
We provide several usage examples in both C++ and Python under [`demos/`](demos/), demonstrating how to utilize the library for various similarity search tasks.

<!--

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



## Running the test suite
```bash
cd build

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
-->

## About
DaiSy is developed by the [diNo research group at LIPADE, Université Paris Cité](https://dino.mi.parisdescartes.fr/). 
It is provided with no warranty, and we encourage contributions from the community to enhance its capabilities and performance. For questions, issues, or contributions, please open an issue or submit a pull request on GitHub.
DaiSy licensed under the [MIT License](LICENSE).














