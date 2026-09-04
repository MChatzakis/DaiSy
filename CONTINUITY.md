# CONTINUITY.md

This file captures the current shape of the DaiSy repository so future contributors and LLM agents can recover context quickly.
It is intentionally operational, not aspirational.

## Current Repo Snapshot

DaiSy is an alpha-stage exact similarity-search library with:

- C++ core implementations
- Optional Python bindings
- Tests, demos, and benchmarks as separate build layers
- Multiple algorithm families spanning in-memory, disk-based, distributed, GPU, and streaming cases

The repo is broad, but integration is still somewhat manual. Many additions require touching several disconnected files.

## High-Level Layout

- `lib/algos/`: algorithm implementations and the main extension point
- `lib/distance_computers/`: distance logic shared by algorithms
- `lib/isax/`: SAX/iSAX support used by several index-based algorithms
- `lib/ds_tree/`: tree structures used by `Hercules`
- `commons/`: shared loaders, params, test and benchmark helpers
- `pybinds/setup.cpp`: Python binding entry point
- `tests/`: C++ tests
- `benchmark/`: benchmark executables
- `demos/`: C++ and Python usage examples
- `docs/`: contributor and architecture docs
- `scripts/groundtruth/`: utilities to generate or validate ground-truth data

## Public And Internal Integration Points

When adding or changing an algorithm, the important wiring points are:

- Base interface: `lib/algos/SimilaritySearchAlgorithm.hpp`
- Source registration: `lib/algos/CMakeLists.txt`
- Public umbrella include: `lib/daisy.hpp`
- Python exposure: `pybinds/setup.cpp`
- Tests: `tests/CMakeLists.txt` plus new `tests/test_*.cpp`
- Benchmarks: `benchmark/CMakeLists.txt` plus new `benchmark/bm_*.cpp`
- Demos: `demos/CMakeLists.txt` plus new `demos/demo_*`

The project does not currently centralize algorithm metadata in one place.
Assume cross-file manual registration is required.

## Algorithm Surface Today

From the current source tree and public includes, the main algorithm set includes:

- `Bruteforce`
- `LbBruteforce`
- `Coconut`
- `Messi`
- `ParIS`
- `Sing`
- `Odyssey`
- `Sofa`
- `Hercules`
- `DumpyOS`
- `Fresh`

These algorithms do not all share the same dependency profile or feature set.

## Important Behavioral Conventions

- Top-k search is the baseline capability.
- Range search is modeled through `SearchConfig` and is not universally implemented.
- Streaming insert is implemented by `BruteForceSearch`, `LbBruteforce`, and `Coconut`;
  the base implementation still throws for algorithms that do not support it.
- Bruteforce streaming grows the owned in-memory database incrementally. LbBruteforce also
  computes SAX summaries incrementally, using the breakpoint set fixed by the initial build.
- `setNormalized(bool)` is a declaration about the input data, not a preprocessing step.
- Data can come from in-memory arrays or file-backed sources via `DataSource`.

If you extend behavior, keep these semantics stable unless you are intentionally redesigning the interface.

## Build System Reality

The root `CMakeLists.txt` drives feature toggles and dependency detection.

Important options:

- `BUILD_PYTHON`
- `BUILD_BENCHMARK`
- `BUILD_TESTS`
- `BUILD_DEMO`
- `BUILD_ODYSSEY`
- `BUILD_SING`
- `BUILD_SOFA`
- `BUILD_COCONUT`

Current optional dependency model:

- MPI gates `Odyssey`
- CUDA gates `Sing`
- FFTW3 gates `Sofa`
- Coconut can be fully toggled off

Backward-compatibility option aliases exist for some old names such as MPI and CUDA toggles.

## Known Rough Edges

These are worth checking before assuming the tree is fully uniform:

- `pybinds/setup.cpp` is monolithic and repetitive; new bindings usually require copy-adapt work rather than a clean abstraction.
- Several test entries for DTW are present but commented out in `tests/CMakeLists.txt`.
- Test and benchmark registration is manual and verbose.
- Optional algorithms are not uniformly guarded everywhere at the conceptual level, so verify includes and bindings carefully when adding new conditional code.
- The repo mixes “library architecture” docs with practical contributor docs; not every workflow detail is encoded in one canonical place.

## What To Check Before Adding A New Algorithm

Read these first:

- `docs/how-to-contribute.md`
- `README.md`
- `lib/algos/SimilaritySearchAlgorithm.hpp`
- A similar existing algorithm in `lib/algos/`
- `lib/algos/CMakeLists.txt`
- `pybinds/setup.cpp`
- `tests/CMakeLists.txt`
- `benchmark/CMakeLists.txt`
- `demos/CMakeLists.txt`

Then decide:

- Is the algorithm always buildable, or optional?
- Does it support top-k only, or also range?
- Does it support L2 squared, DTW, or both?
- Does it need new loaders, new shared utilities, or only local code?
- Should it be exposed to Python now, or intentionally remain C++-only?

## Expected Contribution Pattern

For a normal new algorithm contribution, expect to touch at least:

- `lib/algos/<Algo>.hpp`
- `lib/algos/<Algo>.cpp`
- `lib/algos/CMakeLists.txt`
- `lib/daisy.hpp` if it should be part of the public header surface
- `tests/test_<Algo>_*.cpp`
- `tests/CMakeLists.txt`
- `demos/demo_<Algo>_*.cpp`
- Python demo and binding files if supported from Python
- `README.md`

Benchmarks are also expected for performance-oriented algorithms.

## Documentation Expectations

The current repo already has useful support docs:

- `docs/how-to-contribute.md`
- `docs/benchmark-framework-guide.md`
- `docs/test-framework-guide.md`
- `docs/pybinds-guide.md`
- `docs/demos-guide.md`

Use them, but do not assume they fully replace source inspection.
The repository still relies heavily on pattern matching against existing implementations.

## Practical Advice For LLMs

- Do not edit only one layer and assume the job is done.
- Mirror existing algorithm naming exactly across files.
- Prefer the nearest existing algorithm as a template.
- Be explicit about unsupported features.
- Keep optional dependencies optional.
- If you add a build flag, document it immediately.
- If you notice a current inconsistency, record it in docs or fix it in the same change if safe.

## When This File Should Be Updated

Update `CONTINUITY.md` when:

- A new algorithm is added or removed
- The expected contributor workflow changes
- A major build flag or optional dependency changes
- Python binding structure changes materially
- Tests, demos, or benchmarks gain a new required pattern
- A major known rough edge is fixed or a new one appears
