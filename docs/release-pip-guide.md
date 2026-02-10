# DaiSy Pip Release Process

This document outlines the complete process for rebuilding DaiSy from scratch as a pip-ready package. All steps have been tested and verified to work correctly.

## Overview

The goal is to create distribution-ready packages (wheel and source) that can be installed via pip. The process involves cleaning the project, fixing dependencies, restructuring the package layout, and building for distribution.

## Step-by-Step Process

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
            ("ODYSSEY_MPI", "1" if ENABLE_MPI else "0"),  # Enabled if mpi4py is available
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
python -c "import daisy; print('Version:', daisy.__version__); print('Available:', [x for x in dir(daisy) if not x.startswith('_')][:8])"
```

Expected output:
```
Version: 1.0.0
Available: ['BruteForceSearch', 'DTW', 'DistanceType', 'L2_SQUARED', 'LbBruteforce', 'Messi', 'Odyssey', 'ParIS']
```

---

## Available Algorithms

### Default Build (Now Includes Odyssey!)

The pip-ready package includes these algorithms **by default**:

| Algorithm | Type | Distance | Status |
|-----------|------|----------|--------|
| **BruteForceSearch** | Exact | L2, DTW | ✅ Always Available |
| **LbBruteforce** | Exact + LB | L2, DTW | ✅ Always Available |
| **MESSI** | Indexed | L2, DTW | ✅ Always Available |
| **PARIS** | Disk-based | L2, DTW | ✅ Always Available |
| **Odyssey** | Distributed MPI | L2, DTW | ✅ **NEW** - Now Required (mpi4py) |

### Optional Algorithms (Require Additional Setup)

| Algorithm | Requires | Status |
|-----------|----------|--------|
| **SING** | CUDA Toolkit | ❌ Disabled by default |

---

## Dependencies

### Required
- **mpi4py >= 4.0.3** - NOW REQUIRED for Odyssey distributed search
- **numpy >= 2.2.6**
- **pybind11 >= 3.0.0**

### System Requirements for Building
- **MPI Libraries** (OpenMPI or MPICH)
  - Ubuntu/Debian: `sudo apt-get install libopenmpi-dev openmpi-bin`
  - macOS: `brew install open-mpi`
  - Windows: Download from [Microsoft HPC Pack](https://www.microsoft.com/en-us/download/details.aspx?id=100593)
  - Conda: `conda install openmpi`

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
├── daisy-1.0.0-cp312-cp312-linux_x86_64.whl    (1.1 MB)  [Binary wheel with Odyssey]
└── daisy-1.0.0.tar.gz                          (957 KB)  [Source distribution with Odyssey]
```

### What's Now Included By Default

**All installations now include:**
- ✅ BruteForceSearch  
- ✅ LbBruteforce
- ✅ MESSI
- ✅ ParIS
- ✅ **Odyssey** (Distributed MPI-based search) - **NEWLY REQUIRED!**
- ✅ L2-Squared and DTW distance metrics

**Optional (requires additional setup):**
- ❌ SING (requires: CUDA Toolkit)

To enable SING (GPU acceleration), see "Building with CUDA Support" section above.

## Installation for End Users

DaiSy now requires MPI for Odyssey distributed search. Install with:

```bash
# Ubuntu/Debian: Install MPI libraries first  
sudo apt-get install libopenmpi-dev openmpi-bin

# macOS: Install MPI via Homebrew
brew install open-mpi

# Conda: Install OpenMPI
conda install openmpi

# Then install DaiSy
# From wheel (fastest - includes Odyssey)
pip install daisy-1.0.0-cp312-cp312-linux_x86_64.whl

# From source (requires compilation)
pip install daisy-1.0.0.tar.gz

# Or from PyPI (once uploaded)
pip install daisy
```

## Important: MPI Requirement

**Starting with version 1.0.0, DaiSy requires MPI libraries to build and run.**

This is because Odyssey (distributed search) is now a required component. When you install the wheel, mpi4py will be automatically installed, but you need MPI libraries on your system:

- **Linux**: `libopenmpi-dev` / `mpich-devel` (system package)
- **macOS**: Install via Homebrew
- **Windows**: Microsoft HPC Pack
- **Conda environment**: `conda install openmpi`

## Quick Reference: One-Command Summary

```bash
# Full rebuild from scratch (with MPI)
cd /home/mchatzakis/diNoSimilaritySearch && \
pip uninstall -y daisy && \
rm -rf build/ dist/ *.egg-info && \
python -m build && \
pip install --force-reinstall --no-deps dist/daisy-1.0.0-cp312-cp312-linux_x86_64.whl && \
python -c "import daisy; print('✓ DaiSy', daisy.__version__, 'installed successfully'); print('✓ Odyssey available:', hasattr(daisy, 'Odyssey'))"
```

## Troubleshooting

If the build fails:

1. **Missing pybind11**: Ensure it's installed: `pip install pybind11>=3.0.0`
2. **Missing numpy**: Install it: `pip install numpy>=2.2.6`
3. **Missing MPI headers**: Install MPI development libraries (see "Important: MPI Requirement" above)
4. **Compiler issues**: Ensure you have a C++17 compatible compiler (GCC 6+, Clang 3.4+, MSVC 2015+)
5. **Stale builds**: Always clean before rebuilding: `rm -rf build/ dist/ *.egg-info`

### Odyssey Not Available

If `daisy.Odyssey` is missing:

1. **Check MPI is installed**: `mpicc --version` or `mpirun --version`
2. **Reinstall MPI libraries** and rebuild
3. **Check mpi4py is available**: `python -c "import mpi4py"`

To verify algorithms are available:

```bash
python -c "import daisy; algs = [x for x in dir(daisy) if not x.startswith('_')]; print('Available:', algs); print('Has Odyssey:', 'Odyssey' in algs)"
```

### Build Fails with "mpi.h: No such file"

The MPI development headers aren't installed on your system.

**Solution**: Install MPI development libraries:
```bash
# Ubuntu/Debian
sudo apt-get install libopenmpi-dev openmpi-bin

# macOS
brew install open-mpi

# Fedora/RHEL
sudo dnf install openmpi-devel

# Conda
conda install openmpi
```

Then rebuild:
```bash
cd /home/mchatzakis/diNoSimilaritySearch
rm -rf build/ dist/ *.egg-info
python -m build
```
