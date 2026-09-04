# How to Contribute: Adding a New Algorithm

This guide walks you through the process of adding a new similarity search algorithm to the DaiSy library. Following these steps will ensure your algorithm integrates properly with the existing codebase and maintains consistency with current practices.

## Our TODO list

DaiSy is currently in active development. We are currently fixing bugs and installation issues. We have a list of future algorithms we plan to implement. 
If you are interested in contributing, please reach out to us to coordinate efforts and avoid duplication of work. 
You may reach us by contacting the main authors of the project, [Francesca Del Gaudio](mailto:francescadelgaudio56@gmail.com) and [Manos Chatzakis](mailto:manos.chatzaki@gmail.com).

Here is a (non-exhaustive) mockup of our future and ongoing goals:

- Extension of DaiSy for subsequence similarity search
- Extension of DaiSy for more algorithms (e.g., SFA, Hercules, Dumpy, etc.)
- Streaming / updatable indexing for more algorithms (currently supported by Bruteforce, LbBruteforce, and Coconut)
- Implementation of a DaiSy autotuner to automatically optimize indexing and search parameters
- Extension of DaiSy to support learned optimization approaches, e.g., LeaFi and ProS

## Prerequisites

Before starting, ensure you have:
- A working development environment with CMake, C++17 or later, and Python 3.7+
- Familiarize yourself with the existing algorithms in `lib/algos/` (e.g., `Bruteforce.hpp/cpp`, `Messi.hpp/cpp`)
- Understanding of the `SimilaritySearchAlgorithm` interface

## Step-by-Step Guide

### 1. Implement the Algorithm in `lib/algos/`

Create your new algorithm under the `lib/algos/` directory, following the existing library structure.

**Requirements:**
- Create a header file (e.g., `YourAlgorithm.hpp`) that inherits from `SimilaritySearchAlgorithm`
- Create an implementation file (e.g., `YourAlgorithm.cpp`)
- Implement all pure virtual methods from the `SimilaritySearchAlgorithm` interface
- Follow the naming and style conventions used by existing algorithms
- Include proper error handling and input validation

**File structure example:**
```
lib/algos/YourAlgorithm.hpp
lib/algos/YourAlgorithm.cpp
```

**CMake integration:**
- Add your algorithm to `lib/algos/CMakeLists.txt` under the appropriate source and header file lists

### 2. Add Tests to the Test Module

Create comprehensive tests for your new algorithm to ensure correctness.

**Requirements:**
- Create test files in the `tests/` directory (e.g., `test_YourAlgorithm_L2Square.cpp`, `test_YourAlgorithm_DTW.cpp`)
- Test with support for different distance metrics (at minimum L2Square and DTW)
- Include tests for edge cases (empty datasets, single queries, k > dataset size, etc.)
- Verify results against known ground truth or brute force results
- Add test entries to `tests/CMakeLists.txt`

**Reference:** Look at existing test files like `test_Messi_L2Square.cpp` for structure and patterns.

### 3. Add to the Benchmarks Module

Create benchmarks to measure performance characteristics of your algorithm.

**Requirements:**
- Create benchmark files in the `benchmark/` directory (e.g., `bm_YourAlgorithm_L2Square.cpp`)
- Include benchmarks for different data characteristics (size, dimensionality, query count)
- Use the existing `bm_utils.hpp` utilities for consistency
- Add entries to `benchmark/CMakeLists.txt`

**Reference:** See `bm_Messi_L2Square.cpp` for an example implementation.

### 4. Add Python Bindings

Integrate your algorithm into the Python interface by modifying `pybinds/setup.cpp`.

**Requirements:**
- Add Python binding definitions using pybind11
- Export the algorithm class and its methods
- Include proper docstrings describing parameters and return types
- Ensure the Python API is consistent with existing algorithms
- Test the Python bindings work correctly

**Pattern to follow:**
```cpp
py::class_<YourAlgorithm>(m, "YourAlgorithm")
    .def(py::init<...>())
    .def("search", &YourAlgorithm::search)
    // ... other methods
```

### 5. Handle Optional Dependencies (if applicable)

If your algorithm has strong dependencies on external libraries or specialized hardware, make compilation optional.

**Requirements:**
- Add CMake flags to control optional compilation (e.g., `ENABLE_YOUR_ALGORITHM`)
- Update `CMakeLists.txt` to conditionally compile your algorithm
- Provide clear documentation about required dependencies
- Ensure the build succeeds with the option disabled
- Test that other algorithms still work when your algorithm is disabled

**Example CMake pattern:**
```cmake
option(ENABLE_YOUR_ALGORITHM "Build with YourAlgorithm support" ON)
if(ENABLE_YOUR_ALGORITHM)
    # Add your algorithm source files and dependencies
endif()
```

### 6. Run All Tests

Verify that your implementation is correct and doesn't break existing functionality.

**Requirements:**
- Run the full test suite: `ctest` in the `build/` directory
- Ensure all existing tests still pass
- Ensure all new tests for your algorithm pass
- Fix any compilation warnings specific to your code
- Test on different platforms if possible (Linux, macOS, Windows)

**Command:**
```bash
cd build/
cmake ..
make
ctest --verbose
```

### 7. Create Demonstration Programs

Provide clear examples of how to use your algorithm in both Python and C++.

**Requirements:**
- Create C++ demo: `demos/demo_YourAlgorithm_L2Square.cpp` and `demos/demo_YourAlgorithm_DTW.cpp`
- Create Python demo: `demos/demo_YourAlgorithm_L2Square.py` and `demos/demo_YourAlgorithm_DTW.py`
- Include comments explaining the key steps
- Demonstrate how to:
  - Instantiate the algorithm
  - Load/prepare data
  - Perform queries
  - Interpret results
- Add entries to `demos/CMakeLists.txt` for C++ demos

**Reference:** Look at `demo_Messi_L2Square.cpp` and `demo_Messi_L2Square.py` for structure.

### 8. Update Documentation

Keep the project documentation current and comprehensive.

**Requirements:**
- Update `README.md` to mention your new algorithm
- Add a section in the README describing:
  - Algorithm name and key characteristics
  - Time and space complexity
  - When to use this algorithm
  - Any limitations or special requirements
- Update `docs/` files if there are architecture-specific considerations:
  - `daisy-architecture.puml` (if applicable)
  - `daisy-components.puml` (if applicable)
- Add docstrings to your C++ code and Python bindings
- Consider adding a separate documentation page if the algorithm is complex

### 9. Rebuild the Pip Package

After communicating with the maintainers and receiving approval, the pip package can be rebuilt and released.

**Requirements:**
- Ensure all previous steps have been completed and reviewed
- Coordinate with project maintainers before releasing
- Update version numbers in `pyproject.toml` and `setup.py` if necessary
- Build the distribution: `python setup.py sdist bdist_wheel`
- Test the pip package installation and functionality
- Upload to PyPI via the release process

**Reference:** See `docs/release-pip-guide.md` for detailed release instructions.

## Checklist

Use this checklist to ensure you haven't missed anything:

- [ ] Algorithm implementation in `lib/algos/` (header + implementation)
- [ ] Algorithm added to `lib/algos/CMakeLists.txt`
- [ ] Unit tests created and passing
- [ ] Tests added to `tests/CMakeLists.txt`
- [ ] Benchmarks created for performance evaluation
- [ ] Benchmarks added to `benchmark/CMakeLists.txt`
- [ ] Python bindings in `pybinds/setup.cpp`
- [ ] Optional dependencies handled (if applicable)
- [ ] CMake flags added (if applicable)
- [ ] All tests pass (`ctest --verbose`)
- [ ] C++ and Python demo programs created
- [ ] Demos added to `demos/CMakeLists.txt`
- [ ] README.md updated
- [ ] Other documentation updated
- [ ] Code reviewed by maintainers
- [ ] Pip package rebuilt (post-approval)
