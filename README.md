## Requirements

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
