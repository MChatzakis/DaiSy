# Installation Guide for diNo-lib

## For Users: Installing the Package

### Quick Install (Recommended)

```bash
# Create and activate virtual environment (recommended)
python -m venv diNo_env
source diNo_env/bin/activate  # On Windows: diNo_env\Scripts\activate

# Install from source
git clone https://github.com/MChatzakis/diNo-lib.git
cd diNo-lib
git submodule update --init --recursive

# Install build dependencies and package
pip install pybind11 numpy setuptools wheel
pip install .

# Or with optional dependencies
pip install .[mpi]  # With MPI support
pip install .[dev]  # With development tools
```

## For Developers: Development Setup

### Prerequisites

```bash
# Make sure you have Python 3.8+ and pip installed
python --version
pip --version
```

### Development Environment Setup

```bash
# Create virtual environment for development
python -m venv diNo_dev_env
source diNo_dev_env/bin/activate  # On Windows: diNo_dev_env\Scripts\activate

# Clone repository
git clone https://github.com/MChatzakis/diNo-lib.git
cd diNo-lib
git submodule update --init --recursive

# Install build dependencies
pip install pybind11 numpy setuptools wheel build

# Install in development mode (editable install)
pip install -e .[dev]
```

## For Maintainers: Creating Distribution Packages

### Build Packages

```bash
# Create clean virtual environment for packaging
python -m venv packaging_env
source packaging_env/bin/activate

# Install packaging tools
pip install build twine pybind11 numpy

# Build source distribution and wheel
python -m build

# This creates files in dist/ directory:
# - dino-lib-1.0.0.tar.gz (source distribution)
# - dino-lib-1.0.0-py3-none-any.whl (wheel distribution)
```

### Test Built Package

```bash
# Create fresh test environment
python -m venv test_env
source test_env/bin/activate

# Install from built wheel
pip install dist/dino-lib-1.0.0-*.whl

# Test the installation
python -c "import diNoSimilaritySearch; print('Installation successful!')"
```

### Upload to PyPI (when ready)

```bash
# Upload to TestPyPI first
twine upload --repository testpypi dist/*

# Upload to PyPI
twine upload dist/*
```

## Usage

```python
import numpy as np
from diNoSimilaritySearch import DistanceType, BruteForceSearch

# Create random data
np.random.seed(100)
db = np.random.randn(1000, 64).astype(np.float32)
query = np.random.randn(10, 64).astype(np.float32)

# Create search index
index = BruteForceSearch(DistanceType.L2_SQUARED)
index.buildIndex(db)

# Search
I, D = index.searchIndex(query, k=5)
print("Indices:", I)
print("Distances:", D)
```

## Troubleshooting

- If you get import errors, make sure pybind11 and numpy are installed
- On some systems you may need to install a C++ compiler (gcc/clang)
- For CUDA support, make sure CUDA toolkit is installed
- For MPI support, install MPI library (e.g., `sudo apt-get install libopenmpi-dev`)
