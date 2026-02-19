<p align="center">
<img src="https://github.com/MChatzakis/DaiSy/blob/main/assets/logo.png" alt="DaiSy Logo" height="280"> 
</p>

<h1 align="center">DaiSy</h1>
<h2 align="center">A Library for Scalable Data Series Similarity Search</h2>

<p align="center">
<a href="https://github.com/MChatzakis/daisy/stargazers"><img src="https://img.shields.io/github/stars/MChatzakis/daisy?style=social" alt="GitHub Stars"></a>
</p>

<h4 align="center">Francesca Del Gaudio, Manos Chatzakis, Gayathiri Ravendirane, Botao Peng, Themis Palpanas</h4>

Exact similarity search over large collections of data series is a fundamental operation in modern applications, yet existing solutions are often fragmented, specialized, or tailored to specific execution environments.
We present DaiSy (**Da**ta series s**i**milarity **S**earch librar**y**), a unified library for exact data series similarity search that integrates multiple state-of-the-art algorithms within a single, coherent framework.
DaiSy is the first library to support exact similarity search across diverse execution environments, including implementations for disk-based, in-memory, GPU-accelerated, and distributed scalable similarity search.
The library supports interfaces in both C++ and Python, enabling, researchers and practitioners to easily integrate its functionality in a variety of tasks. 

**<span style="color:red">ALPHA VERSION</span>**: Currently, DaiSy is experimental. The library is still under active development. We welcome suggestions and bug reports.

<!--When using DaiSy, please consider citing the following paper:

```latex
Coming Soon!
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
| `BUILD_PYTHON` | Enable Python bindings | `OFF` | Python 3.10+ |
| `BUILD_BENCHMARK` | Build benchmarking tools | `OFF` | GoogleBenchmark |
| `BUILD_TESTS` | Build test suite | `OFF` | GoogleTest |
| `BUILD_DEMO` | Build demonstration applications | `ON` | Core library |
| `BUILD_ODYSSEY` | Enable MPI for distributed computing | `OFF` | OpenMPI/MPICH |
| `BUILD_SING` | Enable CUDA for GPU acceleration | `OFF` | CUDA Toolkit |
| `DEBUG_MSG` | Enable debug output | `OFF` | None |

To compile:
```bash
mkdir build && cd build

cmake ..
make
```

### DaiSy with Python
If you intent to use only the Python interface, you can install the library directly from PyPI using pip:
```bash
pip install daisy-exact-search
```

If you want to use Odyssey, you will need to install mpi:
```bash
pip install daisy-exact-search[mpi]
```


### Compatibility issues
Kindly note that we are aware for compatibility issues related to ARM processors (e.g., Apple MX processors). 
Due to pthread-barriers and SIMD being unavailable on ARM, we currently noticing compilations failling on ARM machines.
We are currently working on possible solutions, however we recommend using DaiSy on non-ARM machines for the time being.

<!--
```bash
python3.12 -m venv DaiSy_env

source DaiSy_env/bin/activate 
pip install -r requirements_DaiSy.txt
```
-->


### Others
We provide several usage examples in both C++ and Python under [`demos/`](demos/), demonstrating how to utilize the library for various similarity search tasks.
We provide several troubleshooting guides and extra resources in the [`docs/`](docs/) directory. 
In this directory, we also provide useful information about how to contribute to the project, and how to implement new algorithms.



<!--

#### Build con CUDA (per testare la demo Sing)
Richiede **CUDA Toolkit** installato (`nvcc --version` e `nvidia-smi` funzionanti). La demo `demo_Sing_L2Square` viene compilata solo se CUDA è disponibile.

```bash
mkdir -p build && cd build
# Se non usi MPI (consigliato per test locali):
cmake .. -DBUILD_ODYSSEY=OFF -DBUILD_SING=ON -DBUILD_DEMO=ON
# Oppure con MPI:
# cmake .. -DBUILD_SING=ON -DBUILD_DEMO=ON

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
This work was supported by the Hellenic Foundation for Research and Innovation (HFRI) under the “Second Call for HFRI Research Projects to support Faculty Members and Researchers” (project number: 3684).

<!--DaiSy is developed by researchers at the [diNo research group, LIPADE, Université Paris Cité](https://dino.mi.parisdescartes.fr/). 

It is provided with no warranty, and we encourage contributions from the community to enhance its capabilities and performance. For questions, issues, or contributions, please open an issue or submit a pull request on GitHub.
DaiSy licensed under the [MIT License](LICENSE).

For questions and suggestions through mail, you can contact us at [manos.chatzaki@gmail.com](mailto:manos.chatzaki@gmail.com).-->

The logo of DaiSy was designed by [Eva Chamilaki](https://evachamilaki.github.io).












