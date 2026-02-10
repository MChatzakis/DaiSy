# Benchmark Module

This directory contains the benchmarking framework for DaiSy, providing performance evaluation tools for various similarity search algorithms on time series data.

## Components

### Core Benchmarking Framework

- **[bm_utils.hpp/cpp](../benchmark/bm_utils.hpp)**: Contains the main benchmarking infrastructure
  - `runSSTBenchmark()`: Core function that loads data, builds indices, and measures search performance
  - Timer management for precise performance measurements
  - Memory usage tracking capabilities
  - Result formatting and output utilities

### Algorithm-Specific Benchmarks

Each benchmark executable follows a consistent pattern:

1. **Data Loading**: Load database and query datasets from binary files
2. **Index Building**: Construct the search index for the specific algorithm
3. **Search Execution**: Perform k-nearest neighbor searches on queries
4. **Performance Measurement**: Record timing and memory usage statistics

## Usage

Benchmarks are built here: [benchmark](../benchmark/).

### Benchmark Parameters

Benchmarks can be configured through command-line arguments or configuration files:

- **Dataset paths**: Specify database and query file locations
- **Thread count**: Number of parallel threads for computation
- **k-value**: Number of nearest neighbors to retrieve
- **Algorithm parameters**: Algorithm-specific tuning parameters

### Expected Input Format

Benchmarks expect binary data files in the following format:

- Database files: `*.data.*.bin` (time series database)
- Query files: `*.query.*.bin` (query time series)
- Data format: 32-bit float arrays, row-major order

## Performance Metrics

The benchmarking framework measures:

1. **Index Build Time**: Time required to construct the search index
2. **Search Time**: Time for k-NN query execution
3. **Memory Usage**: Peak memory consumption during operations
4. **Throughput**: Queries processed per second
5. **Scalability**: Performance across different thread counts
