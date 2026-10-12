# Tests Module

This directory contains the comprehensive testing suite for the DaiSy library, including unit tests, integration tests, performance regression tests, and correctness validation for all similarity search algorithms.

See [tests](../tests/) for details.

## Testing Framework

[test_utils.hpp/cpp](../tests/test_utils.hpp): Provides common utilities and infrastructure used across all test modules.

## How to Add a Test

### 1. Create a New Test File

### 2. Include Required Headers

### 3. Define Test Structure

Follow the parameterized test pattern used throughout the test suite for BruteForce_L2Square.cpp

### 4. Add Test to CMakeLists.txt

### 5. Build and Run

### 6. Test Requirements

- Tests must use the `runSST()` function from `test_utils.hpp` for consistency
- Tests should be parameterized to run with different configurations (k-values, thread counts, datasets)
- Include proper ground truth validation
- Follow the existing naming conventions and code structure

## MESSI exact top-k regressions

`test_Messi_TopK_Distinct` compares every returned neighbor against exhaustive
brute-force ground truth, checks distinct IDs, and recomputes the squared-L2
distance of each returned vector. Missing neighbors are failures; a recall
threshold below 100% is not sufficient for exact search. Integer-distance tie
fixtures allow any distinct IDs with the exact expected distances.

The parameterized cases cover Gaussian and adaptive breakpoints, one and four
query workers, k=1/10/100/1000, repeated queries, and streaming updates. The
inclusive-range regression checks zero and nonzero radii, exact boundary hits and
partial sums equal to the radius before later nonzero chunks. The adaptive MESSI
case in `test_equidepth` checks all k results with a small floating-point
tolerance rather than only checking finite output. Simple CI
runs these regressions through its complete CTest suite.

`test_sax_breakpoints` covers positive/negative shifted cuts, repeated cuts, exact
boundaries and both tail bins. It also checks scalar and SIMD record lower bounds
against exact squared L2. The tests fail with the old sign-based SAX fallback.

With `BUILD_PYTHON=ON`, `MessiPythonLayouts` additionally checks exact top-k and
inclusive range results for contiguous, `.fvecs`-style, row-strided, Fortran and
negative-stride queries, before and after streaming inserts. NumPy must be
installed for this Python test. Python builds with Coconut disabled omit its binding.

For the original unnormalized SIFT1M data, build Python bindings and install
`numpy` and `faiss-cpu`, then run the optional manual audit:

```bash
PYTHONPATH=. python tests/check_messi_sift.py --dataset /path/to/sift --output sift-report.json
```

The audit uses 100 queries, k=1/10/100/1000, Gaussian/adaptive modes, one/four
search workers, repeated queries, static 20K/1M builds and streaming 750K→875K→1M.
It checks every distance against exact FAISS IndexFlatL2, recomputes returned
ID/distance pairs, rejects duplicates and validates the full-dataset reference
against the published SIFT ground truth. Distances of raw integer descriptors
are compared exactly; equally distant alternative IDs are allowed. The dataset
and FAISS are not required for the regular C++ tests.
