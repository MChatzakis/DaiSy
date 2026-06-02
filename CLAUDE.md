# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

DaiSy is a C++17/Python library for **exact** k-nearest-neighbor similarity search over data series. It unifies multiple state-of-the-art algorithms under one API. The core is compiled C++; Python access is via a pybind11 extension (`daisy._core`).

> **ARM/Apple Silicon note**: pthread-barriers and SIMD intrinsics are unavailable on ARM. Compilation currently fails on Apple M-series chips. Use a non-ARM machine for development.

## Build (C++)

```bash
git submodule update --init --recursive   # required once after clone

mkdir build && cd build
cmake ..           # defaults: BUILD_DEMO=ON, all others OFF
make -j$(nproc)
```

Key CMake flags:

| Flag | Purpose |
|------|---------|
| `-DBUILD_TESTS=ON` | GoogleTest suite |
| `-DBUILD_BENCHMARK=ON` | GoogleBenchmark executables |
| `-DBUILD_PYTHON=ON` | pybind11 extension (`daisy/_core.*.so`) |
| `-DBUILD_ODYSSEY=ON` | MPI-distributed Odyssey (needs OpenMPI/MPICH) |
| `-DBUILD_SING=ON` | CUDA-accelerated Sing (needs CUDA Toolkit; default arch=75) |
| `-DDEBUG_MSG=ON` | Verbose CMake + runtime output |

## Run Tests

```bash
cd build
cmake .. -DBUILD_TESTS=ON && make -j$(nproc)
ctest --output-on-failure --verbose

# Run a single test binary
./tests/test_Messi_L2Square
```

## Run Benchmarks

```bash
cd build
cmake .. -DBUILD_BENCHMARK=ON && make -j$(nproc)
./benchmark/bm_Messi_L2Square
```

## Python

```bash
# Install from PyPI
pip install daisy-exact-search

# Build extension from source (development)
pip install -e ".[dev]"

# Or build via CMake and import in-place
cmake .. -DBUILD_PYTHON=ON && make -j$(nproc)
# _core.*.so lands in daisy/ and is importable directly
```

Demos under `demos/` (`demo_Messi_L2Square.py`, etc.) are the canonical usage examples.

## Architecture

### Core abstraction (`lib/`)

```
SimilaritySearchAlgorithm   (lib/algos/SimilaritySearchAlgorithm.hpp)
    ├── buildIndex(DataSource*)      ← pure virtual
    └── searchIndex(query, n, k, I, D)  ← pure virtual
```

All six algorithms (`BruteForceSearch`, `LbBruteforce`, `Messi`, `ParIS`, `Sing`, `Odyssey`) inherit from this base. A caller always calls `buildIndex` then `searchIndex`; both accept float32 row-major arrays.

**DataSource** (`lib/algos/DataSource.hpp`) abstracts input: `InMemoryDataSource` wraps a raw `float*`, `FileDataSource` reads a binary flat file. ParIS and Odyssey require `FileDataSource` (disk-based algorithms).

**DistanceComputer** (`lib/distance_computers/DistanceComputer.hpp`) provides `L2_SQUARED` and `DTW` with SIMD acceleration. LB_Keogh lower bounds are used by `LbBruteforce` and `Messi` for pruning.

**iSAX index** (`lib/isax/`) implements the Symbolic Aggregate approXimation index used internally by `LbBruteforce`, `Messi`, and `ParIS`.

### Python layer (`pybinds/setup.cpp` → `daisy/_core`)

`pybinds/setup.cpp` contains all pybind11 class definitions. Each algorithm class is bound individually; `buildIndex` accepts a 2D `float32` numpy array (or a file path for ParIS). `searchIndex` returns `(indices: ndarray[uint64], distances: ndarray[float32])` both shaped `(n_query, k)`.

Odyssey's Python binding writes the numpy array to `/tmp/odyssey_pybind_db.bin` before building the index—this is intentional (Odyssey is disk-based).

### Supporting directories

- `commons/` — shared C++ utilities: data loaders (`dataloaders`), CLI parameter parsing (`paramSetup`), general helpers (`common`)
- `demos/` — one `.cpp` + one `.py` per algorithm × distance metric
- `tests/` — GoogleTest; all tests use `runSST()` from `test_utils.hpp` and compare against ground truth in `tests/groundtruth/`
- `benchmark/` — GoogleBenchmark; uses `bm_utils.hpp`
- `extern/` — git submodule: pybind11

## Adding a New Algorithm

1. `lib/algos/NewAlgo.hpp/.cpp` — inherit `SimilaritySearchAlgorithm`, implement `buildIndex` and `searchIndex`
2. Add to `lib/algos/CMakeLists.txt`
3. Add pybind11 binding block in `pybinds/setup.cpp` (follow the `Messi` block as a template)
4. Add tests in `tests/test_NewAlgo_L2Square.cpp` and `_DTW.cpp`; register in `tests/CMakeLists.txt`
5. Add benchmarks in `benchmark/bm_NewAlgo_L2Square.cpp`; register in `benchmark/CMakeLists.txt`
6. Add C++ and Python demos in `demos/`
7. Update `README.md` and `lib/daisy.hpp` (the umbrella include)

If the algorithm requires an optional hardware dependency (GPU, MPI), guard it with a CMake flag and `#ifdef` following the `BUILD_SING` / `BUILD_ODYSSEY` pattern.
