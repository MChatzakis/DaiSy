## Requirements

- Python 3.8.16 or Python 3.9.17 or Python 3.10.13 ← recommended
- GoogleTest requires at least C++14

## Pip

```bash
pip install -r requirements.txt
```

## Conda

Note: change the `my_custom_env` in .yml to your env name.

```bash
conda env create -f environment.yml
```

## CMake

```bash
mkdir build
cd build
cmake ..
cmake --build .
```

## CTEST

```bash
cmake -S . -B build
cmake --build build
ctest --test-dir build --output-on-failure
```
