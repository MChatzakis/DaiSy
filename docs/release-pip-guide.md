# DaiSy Pip Release Process

This document outlines the complete process for rebuilding DaiSy from scratch as a pip-ready package. All steps have been tested and verified to work correctly.

## Overview

The goal is to create distribution-ready packages (wheel and source) that can be installed via pip. The process involves cleaning the project, fixing dependencies, restructuring the package layout, and building for distribution.

## Step-by-Step Process

> **Note:** The steps below describe the conceptual approach. The actual implementation in setup.py version 1.0.0+ includes additional platform detection to disable incompatible compiler flags on macOS and Windows. See the "Platform Compatibility" section above for details.

### 1. Clean All Previous Build Artifacts

Remove all existing build outputs and installation artifacts:

```bash
cd /home/mchatzakis/diNoSimilaritySearch
pip uninstall -y daisy
rm -rf build/ dist/ *.egg-info
```

**Why:** Ensures a clean slate and avoids conflicts from previous builds.

---

### 2. Fix gtest Dependency Issue

The `commons/test_bm_utils.cpp` file includes gtest headers which are not available in the pip environment. Remove this file from the build sources in `setup.py`:

Edit `setup.py` and remove the line:
```python
"commons/test_bm_utils.cpp",
```

from the sources list (line ~41).

**Why:** The test utilities are not needed for the Python package and cause undefined symbol errors.

---

### 3. Restructure Package Layout

Create a proper Python package structure:

```bash
mkdir -p daisy
mv daisy/__init__.py daisy/__init__.py  # Move existing init file into daisy package
```

**Why:** Allows proper Python package discovery and namespace management.

---

### 4. Update Python Package Initialization

Edit `daisy/__init__.py` to import from the C++ backend:

```python
from ._core import *
```

instead of:

```python
from .daisy import *
```

**Why:** The C++ extension is now named `_core` to separate the Python wrapper from the compiled module.

---

### 5. Rename C++ Module Export

Edit `pybinds/setup.cpp` and change the module definition:

```cpp
PYBIND11_MODULE(_core, m)
```

instead of:

```cpp
PYBIND11_MODULE(daisy, m)
```

**Why:** Creates the extension as `daisy._core`, allowing the `daisy/__init__.py` to wrap it cleanly.

---

### 6. Update setup.py Configuration

Make the following changes to `setup.py`:

#### 6a. Check for optional MPI support
```python
# Check for optional MPI support
try:
    import mpi4py
    ENABLE_MPI = True
except ImportError:
    ENABLE_MPI = False
```

#### 6b. Import find_packages
```python
from setuptools import find_packages
```

#### 6c. Update extension sources - Add Odyssey and SING
Include all algorithm sources including Odyssey (for MPI) and SING (for CUDA):

```python
sources = [
    "pybinds/setup.cpp",
    # commons
    "commons/common.cpp",
    "commons/dataloaders.cpp",
    "commons/paramSetup.cpp",
    # lib components...
    # lib - algos
    "lib/algos/Bruteforce.cpp",
    "lib/algos/LbBruteforce.cpp",
    "lib/algos/Messi.cpp",
    "lib/algos/ParIS.cpp",
    "lib/algos/Sing.cpp",  # GPU-accelerated search (requires CUDA)
]

# Add Odyssey sources if MPI is available
if ENABLE_MPI:
    sources.extend([
        # Odyssey - MPI-based distributed search
        "lib/algos/odyssey/Odyssey.cpp",
        "lib/algos/odyssey/bsf_sharing.cpp",
        "lib/algos/odyssey/indexing.cpp",
        "lib/algos/odyssey/replication.cpp",
        "lib/algos/odyssey/workstealing.cpp",
    ])
```

#### 6d. Add custom build_ext class
```python
from setuptools.command.build_ext import build_ext as pybind_build_ext

class build_ext(pybind_build_ext):
    """Custom build_ext that ensures daisy/__init__.py is in the lib directory"""
    def build_extensions(self):
        super().build_extensions()
        # After building extensions, copy __init__.py to the build lib directory
        src_file = Path("daisy") / "__init__.py"
        if src_file.exists():
            build_lib = Path(self.build_lib)
            dst_file = build_lib / "daisy" / "__init__.py"
            dst_file.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(str(src_file), str(dst_file))
```

#### 6e. Add custom build_py class
```python
class build_py(_build_py):
    """Custom build_py to ensure daisy/__init__.py is included"""
    def run(self):
        super().run()
        # Ensure it's in the build lib
        src_file = Path("daisy") / "__init__.py"
        if src_file.exists():
            build_lib = Path(self.build_lib)
            dst_file = build_lib / "daisy" / "__init__.py"
            dst_file.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(str(src_file), str(dst_file))
```

#### 6f. Configure extension with conditional MPI/CUDA support
```python
ext_modules = [
    Pybind11Extension(
        "daisy._core",
        sources,
        include_dirs=[...],
        cxx_std=17,
        define_macros=[
            ("VERSION_INFO", '"' + __version__ + '"'),
            ("BUILD_ODYSSEY", "1" if ENABLE_MPI else "0"),  # Enabled if mpi4py is available
            ("SING_CUDA_ENABLED", "0"),  # CUDA disabled by default (optional feature)
        ],
        extra_compile_args=["-fopenmp", "-mavx", "-march=native"],
        extra_link_args=["-fopenmp"],
    ),
]
```

#### 6g. Update setup() call
```python
setup(
    name="daisy",
    ...
    packages=find_packages(),
    include_package_data=True,
    ...
    cmdclass={"build_ext": build_ext_class, "build_py": build_py} if build_ext_class else {"build_py": build_py},
    ...
)
```

**Why:** 
- Includes Odyssey and SING source files for complete algorithm support
- Odyssey is conditionally compiled only if mpi4py is available
- SING requires CUDA which is optional (see "Building with CUDA" section below)

---

### 7. Update MANIFEST.in

Add the daisy package directory to the manifest:

```bash
# Edit MANIFEST.in and add:
recursive-include daisy *.py
```

**Why:** Ensures Python files in the daisy package are included in source distributions.

---

### 8. Clean Again (Important)

Before creating final distributions, clean once more:

```bash
cd /home/mchatzakis/diNoSimilaritySearch
pip uninstall -y daisy
rm -rf build/ dist/ *.egg-info
```

---

### 9. Build Distribution Packages

Build both wheel (binary) and source distributions:

```bash
python -m build
```

**What this does:**
- Compiles the C++ extension (`daisy._core`)
- Creates the Python package structure
- Generates `daisy-1.0.0-cp312-cp312-linux_x86_64.whl` (binary wheel)
- Generates `daisy-1.0.0.tar.gz` (source distribution)

Both files are placed in the `dist/` directory.

---

### 10. Verify Installation

Test the wheel package installation:

```bash
cd /tmp
pip install --force-reinstall --no-deps /home/mchatzakis/diNoSimilaritySearch/dist/daisy-1.0.0-cp312-cp312-linux_x86_64.whl
```

---

### 11. Test Functionality

Verify the package works correctly:

```bash
python -c "import daisy; print('Version:', daisy.__version__); algs = [x for x in dir(daisy) if not x.startswith('_')]; print('Algorithms:', algs[:10])"
```

**Expected output on Linux with MPI:**
```
Version: 1.0.0
Algorithms: ['BruteForceSearch', 'DTW', 'DistanceType', 'L2_SQUARED', 'LbBruteforce', 'Messi', 'Odyssey', 'ParIS']
```

**Expected output on macOS (without MPI):**
```
Version: 1.0.0
Algorithms: ['BruteForceSearch', 'DTW', 'DistanceType', 'L2_SQUARED', 'LbBruteforce', 'Messi', 'ParIS']
```

---

## Platform Compatibility

### Cross-Platform Build Strategy

Starting with version 1.0.0, DaiSy automatically adapts compilation flags based on the target platform:

**Linux (x86_64):**
- ✅ Full OpenMP support (`-fopenmp`)
- ✅ SIMD optimizations (`-mavx -march=native`)
- ✅ MPI/Odyssey supported (if mpi4py installed)
- ✅ All algorithms available

**macOS (Intel & ARM64):**
- ⚠️ No `-fopenmp` (clang limitation; uses std::thread instead)
- ⚠️ No `-mavx` on ARM64 (not supported; x86-64 uses native flags)
- ❌ MPI/Odyssey disabled by default (use optional[mpi] if needed)
- ✅ Core algorithms available: BruteForceSearch, LbBruteforce, MESSI, ParIS

**Windows:**
- ⚠️ No OpenMP external dependency (uses MSVC parallel)
- ❌ MPI/Odyssey disabled by default
- ✅ Core algorithms available: BruteForceSearch, LbBruteforce, MESSI, ParIS

---

## Available Algorithms

### Default Build (Cross-Platform Compatible)

The pip-ready package includes these algorithms **by default on all platforms**:

| Algorithm | Type | Distance | Availability |
|-----------|------|----------|--------|
| **BruteForceSearch** | Exact | L2, DTW | ✅ All platforms |
| **LbBruteforce** | Exact + LB | L2, DTW | ✅ All platforms |
| **MESSI** | Indexed | L2, DTW | ✅ All platforms |
| **ParIS** | Disk-based | L2, DTW | ✅ All platforms |

### Optional Algorithms

| Algorithm | Requires | Platform Support | Status |
|-----------|----------|-----|--------|
| **Odyssey** | mpi4py + MPI libs | Linux only | ⚠️ Optional (install with `[mpi]`) |
| **SING** | CUDA Toolkit | Linux, macOS, Windows | ❌ Disabled by default |

### Installing with Optional Features

**For Odyssey (Distributed MPI Search) on Linux:**
```bash
pip install daisy-exact-search[mpi]
# Then install MPI libraries:
# Ubuntu/Debian: sudo apt-get install libopenmpi-dev openmpi-bin
# Conda: conda install openmpi
```

**For SING (GPU Search) - requires building from source:**
```bash
pip install daisy-exact-search --no-binary :all:
# Set CUDA_HOME and modify setup.py as described below
```

---

## Dependencies

### Required (All Platforms)
- **numpy >= 2.2.6**
- **pybind11 >= 3.0.0**

### Optional
- **mpi4py >= 4.0.3** - For Odyssey distributed search (Linux only)
  - Install with: `pip install daisy-exact-search[mpi]`

### System Requirements for Building
- **C++17 compatible compiler:**
  - Linux: GCC 5.3+ or Clang 3.4+
  - macOS: Xcode 9.1+ (includes Clang)
  - Windows: MSVC 2015+

- **MPI Libraries** (only needed for Odyssey on Linux):
  - Ubuntu/Debian: `sudo apt-get install libopenmpi-dev openmpi-bin`
  - Conda: `conda install openmpi`
  - macOS: `brew install open-mpi` (optional, for manual MPI builds)
  - Windows: Not commonly used with DaiSy

---

## Building with Optional Features

### Building with CUDA Support (for SING)

### Building with CUDA Support (for SING)

To enable the SING GPU-accelerated algorithm:

```bash
# Ensure you have NVIDIA CUDA Toolkit installed
# https://developer.nvidia.com/cuda-downloads

# Set CUDA_HOME environment variable (example for CUDA 12.0)
export CUDA_HOME=/usr/local/cuda-12.0

# Modify setup.py to enable CUDA:
# Change: ("SING_CUDA_ENABLED", "0")
# To:     ("SING_CUDA_ENABLED", "1")

# Rebuild daisy
cd /home/mchatzakis/diNoSimilaritySearch
rm -rf build/ dist/ *.egg-info
python -m build
```

After rebuilding with CUDA support, the `Sing` class will be available in the daisy module.

---

## Final Output

The pip-ready packages are located at:

```
/home/mchatzakis/diNoSimilaritySearch/dist/
├── daisy_exact_search-1.0.0-cp312-cp312-manylinux_2_34_x86_64.whl    (4.2 MB)  [Binary wheel, Linux]
└── daisy_exact_search-1.0.0.tar.gz                                   (957 KB)  [Source, all platforms]
```

### What's Included By Default

**All installations (macOS, Linux, Windows) include:**
- ✅ BruteForceSearch  
- ✅ LbBruteforce
- ✅ MESSI
- ✅ ParIS
- ✅ L2-Squared and DTW distance metrics

**Platform-Specific Features:**
- Linux: Full OpenMP + SIMD optimizations
- macOS: Compatible (no OpenMP, no aggressive SIMD on ARM64)
- Windows: Compatible

**Optional (requires `[mpi]` extra):**
- ⚠️ Odyssey (Linux only, requires MPI + mpi4py)
- ❌ SING (requires CUDA Toolkit, needs rebuild from source)

## Installation for End Users

**Basic Installation (Works on macOS, Linux, Windows):**

```bash
# From PyPI
pip install daisy-exact-search

# From wheel (pre-compiled, fastest)
pip install daisy_exact_search-1.0.0-cp312-cp312-manylinux_2_34_x86_64.whl

# From source (requires compilation)
pip install daisy_exact_search-1.0.0.tar.gz
```

**Installation with Optional Odyssey Support (Linux only):**

```bash
# Install with MPI support - on Linux only
pip install daisy-exact-search[mpi]

# Then ensure MPI libraries are installed:
sudo apt-get install libopenmpi-dev openmpi-bin  # Ubuntu/Debian
# or
conda install openmpi  # Conda
```

### Platform Behavior After Installation

**All platforms get:**
- ✅ BruteForceSearch
- ✅ LbBruteforce  
- ✅ MESSI
- ✅ ParIS
- ✅ L2-Squared and DTW distance metrics

**Linux with `[mpi]` extras gets:**
- ✅ **Odyssey** (distributed search via MPI)

**macOS/Windows:**
- ❌ Odyssey not available (MPI not commonly installed)
- ⚠️ OpenMP disabled (uses std::thread; full compatibility)

---

## Important: The Package Name Changed

**Previous name:** `daisy` (now taken on PyPI)
**Current name:** `daisy-exact-search`

Note: The Python import remains `import daisy` - only the pip package name changed.

```bash
# Install
pip install daisy-exact-search

# Use
import daisy
model = daisy.BruteForceSearch(daisy.DistanceType.L2_SQUARED)
```

## Quick Reference: One-Command Summary

**Linux (full build with Odyssey):**
```bash
cd /home/mchatzakis/diNoSimilaritySearch && \
pip uninstall -y daisy-exact-search && \
rm -rf build/ dist/ *.egg-info && \
conda install openmpi &&  # Ensure MPI available \
python -m build && \
pip install --force-reinstall --no-deps dist/daisy_exact_search-*.whl && \
python -c "import daisy; print('✓ DaiSy installed'); print('✓ Odyssey:', hasattr(daisy, 'Odyssey')); print('✓ Algorithms:', len([x for x in dir(daisy) if not x.startswith('_')]))"
```

**macOS (core algorithms only):**
```bash
cd /home/mchatzakis/diNoSimilaritySearch && \
pip uninstall -y daisy-exact-search && \
rm -rf build/ dist/ *.egg-info && \
python -m build && \
pip install --force-reinstall --no-deps dist/daisy_exact_search-*.whl && \
python -c "import daisy; print('✓ DaiSy installed on macOS'); print('✓ Algorithms:', len([x for x in dir(daisy) if not x.startswith('_')]))"
```

## Troubleshooting

### Installation Issues

If `pip install daisy-exact-search` fails:

1. **Missing pybind11**: `pip install pybind11>=3.0.0`
2. **Missing numpy**: `pip install numpy>=2.2.6`
3. **Compiler issues**: Ensure C++17 compiler (GCC 5.3+, Clang 3.4+, MSVC 2015+)
4. **Stale builds**: Always clean: `rm -rf build/ dist/ *.egg-info`
5. **macOS specific**: Ensure Xcode Command Line Tools are installed:
   ```bash
   xcode-select --install
   ```

### Build Failures

**"error: unsupported option '-fopenmp'" on macOS:**
This is expected - DaiSy automatically disables OpenMP on macOS and uses std::thread instead. The build should still succeed.

**"fatal error: mpi.h: No such file or directory" on Linux:**
MPI development headers aren't installed. Install on Linux only:
```bash
# Ubuntu/Debian
sudo apt-get install libopenmpi-dev openmpi-bin

# Fedora/RHEL  
sudo dnf install openmpi-devel

# Conda
conda install openmpi

# Then rebuild
rm -rf build/ dist/ *.egg-info
python -m build
```

Note: This error is expected and harmless on macOS/Windows (Odyssey will be disabled).

### Odyssey Not Available

If `daisy.Odyssey` is missing:

**On macOS/Windows:** This is expected behavior - Odyssey requires MPI which is not commonly installed on these platforms.

**On Linux:** Install MPI and rebuild:
```bash
# Option 1: System package
sudo apt-get install libopenmpi-dev openmpi-bin

# Option 2: Conda
conda install openmpi

# Then rebuild
rm -rf build/ dist/ *.egg-info
python -m build
pip install --force-reinstall --no-deps dist/daisy_exact_search-*.whl
```

To verify available algorithms:
```bash
python -c "import daisy; algs = [x for x in dir(daisy) if not x.startswith('_')]; print('Available algorithms:', algs)"
```

### Package Name vs Import Name

Remember:
- **pip package name:** `daisy-exact-search`
- **Python import name:** `daisy` (unchanged)

So you install with `pip install daisy-exact-search` but import with `import daisy`.
