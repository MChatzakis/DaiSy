# AGENTS.md

This file is for future contributors and LLM agents working inside the DaiSy repository.
Read this before making changes, especially when adding a new algorithm, binding, benchmark, or test.

## Project Intent

DaiSy is a C++ similarity-search library for data series and vectors, with optional Python bindings.
The repository mixes:

- Core algorithm implementations in `lib/algos/`
- Shared utilities and loaders in `commons/`
- Python bindings in `pybinds/setup.cpp`
- C++ tests in `tests/`
- Benchmarks in `benchmark/`
- Demos in `demos/`
- Contributor docs in `docs/`

The main integration goal is consistency across C++, Python, tests, demos, and build flags.
Do not treat a new algorithm as complete if it only compiles in one layer.

## Ground Rules

- Preserve the existing public naming style: algorithm names are user-facing and appear across C++, Python, demos, tests, and docs.
- Prefer matching an existing algorithm pattern instead of inventing a new structure.
- Keep optional or hardware-specific algorithms behind CMake flags.
- Do not assume every algorithm supports every feature:
  - Range search is optional and depends on `SearchConfig`
  - Streaming insert support is currently special-case behavior
  - MPI, CUDA, and FFTW-backed code must remain optional
- Build failures caused by missing optional dependencies are regressions.
- If a change affects algorithm availability, update docs and CMake in the same patch.

## Files To Read First

Before changing algorithm integration, inspect these files:

- `README.md`
- `docs/how-to-contribute.md`
- `lib/algos/SimilaritySearchAlgorithm.hpp`
- `lib/algos/CMakeLists.txt`
- `tests/CMakeLists.txt`
- `benchmark/CMakeLists.txt`
- `demos/CMakeLists.txt`
- `pybinds/setup.cpp`

If the work changes architecture or responsibilities, also inspect:

- `docs/daisy-architecture.puml`
- `docs/daisy-components.puml`

## Adding A New Algorithm

Minimum expected work:

1. Add the implementation in `lib/algos/`.
2. Make it inherit from `SimilaritySearchAlgorithm`.
3. Add it to `lib/algos/CMakeLists.txt`.
4. Expose it through the public surface if appropriate:
   - C++ umbrella header: `lib/daisy.hpp`
   - Python bindings: `pybinds/setup.cpp`
5. Add tests in `tests/`.
6. Add benchmarks in `benchmark/` if the algorithm is performance-relevant.
7. Add demos in `demos/` for the supported modes.
8. Update `README.md` and any relevant docs.

Use an existing algorithm as the nearest template:

- Simple in-memory baseline: `Bruteforce`, `LbBruteforce`
- iSAX-oriented patterns: `Messi`, `Fresh`, `DumpyOS`
- Optional dependency patterns: `Sing`, `Odyssey`, `Sofa`
- Streaming pattern: `Coconut`

## Interface Expectations

The base interface lives in `lib/algos/SimilaritySearchAlgorithm.hpp`.
Important points:

- `buildIndex(DataSource *data_source)` is the core abstract entry point.
- Convenience overloads already exist for in-memory arrays and filenames.
- `searchIndex(query, n_query, k, I, D)` is the required top-k API.
- `searchIndex(..., SearchConfig, vector<vector<idx_t>>&, vector<vector<float>>& )` has a default top-k implementation and throws by default for unsupported range search.
- `insert()` and `insertBatch()` throw by default. Only override if the algorithm truly supports streaming updates.
- `setNormalized()` controls breakpoint mode assumptions for algorithms that use SAX/iSAX behavior. Do not silently normalize user data.

If your algorithm only supports a subset of features, fail explicitly and clearly.

## Build And Dependency Rules

Current important CMake options at the repo root:

- `BUILD_PYTHON`
- `BUILD_BENCHMARK`
- `BUILD_TESTS`
- `BUILD_DEMO`
- `BUILD_ODYSSEY`
- `BUILD_SING`
- `BUILD_SOFA`
- `BUILD_COCONUT`
- `DEBUG_MSG`

Dependency-sensitive features:

- `Odyssey` depends on MPI
- `Sing` depends on CUDA
- `Sofa` depends on FFTW3
- `Coconut` is toggleable via `BUILD_COCONUT`

When adding a dependency-heavy algorithm:

- Introduce a dedicated CMake option if it can be absent on developer machines or CI.
- Keep the default build usable without that dependency unless maintainers explicitly want otherwise.
- Ensure the library still configures when the dependency is missing and the feature is disabled.

## Testing Expectations

At minimum, new algorithms should be validated against known-correct behavior.
Prefer using existing test patterns and ground-truth assets where possible.

Check:

- Correctness versus brute-force or stored ground truth
- L2 squared and DTW when both are supported
- Edge conditions like small datasets, `k == 1`, `k > 1`, and invalid inputs
- Range search behavior if implemented
- Streaming insert behavior if implemented

Do not add an algorithm without at least one executable test path in `tests/`.

## Python Binding Expectations

Python bindings are currently centralized in `pybinds/setup.cpp`.
That file is large and repetitive by design; follow the existing style unless you are intentionally refactoring it.

When exposing a new algorithm:

- Keep constructor and method naming aligned with the C++ class
- Validate NumPy array rank and shape
- Return `(indices, distances)` consistently
- Expose configuration setters/getters only if they are meaningful and stable
- Add docstrings for non-obvious parameters and behavior

If a feature is optional at build time, guard the binding accordingly.

## Demos And Benchmarks

Treat demos as executable usage documentation.
If users would not know how to instantiate or query the algorithm from the code alone, the integration is incomplete.

Benchmarks should:

- Reuse existing benchmark helpers where possible
- Avoid inventing one-off data loading patterns
- Be added only when the feature is actually buildable under current flags

## Documentation Policy

Whenever you add or materially change an algorithm, update:

- `README.md`
- `docs/how-to-contribute.md` if the contributor workflow changed
- Architecture docs if the new code introduces a new subsystem or dependency pattern

If behavior is temporary, experimental, or known to be incomplete, write that down instead of implying full support.

## Safety Checklist Before Finishing

Before closing work, verify:

- CMake knows about the new files
- The algorithm is not orphaned from tests, demos, or bindings
- Optional dependencies are properly guarded
- User-facing names are consistent across C++, Python, docs, and executable targets
- Unsupported features fail explicitly rather than silently misbehaving
- Docs mention major limitations

## Continuity

Also read `CONTINUITY.md`.
That file captures current repo state, active conventions, and known rough edges that are easy to miss when editing blindly.
