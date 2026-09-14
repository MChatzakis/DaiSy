# Demos Module

The demos module provides practical examples of how to use the DaiSy library's algorithms for data series similarity search. 
Each demo illustrates a specific algorithm or specific distance metric. This module includes both C++ and Python implementations for various algorithms and use cases.

Most demos follow the same batch pattern: `buildIndex(...)` once, then `searchIndex(...)`.
**Bruteforce**, **LbBruteforce**, and **Coconut** additionally support streaming through
`insert(...)` and `insertBatch(...)`. See `demo_Bruteforce_Streaming`,
`demo_LbBruteforce_Streaming`, and `demo_Coconut_Streaming` for live-index examples.

## Demo Program Structure

All demos are located in the [`demos/`](../demos/) directory.

## Configuration and Customization

### Parameter Customization

Each demo can be customized by modifying:

- **Dataset Size**: `n_database`, `n_query`
- **Dimensionality**: `dim` (time series length)
- **Search Parameters**: `k` (number of neighbors)
- **Algorithm Parameters**: Thread count, distance metrics, etc.

### Data Sources

Demos can use:

- **Generated Data**: Random or synthetic time series
- **File Data**: Load from binary files in `data/` directory
- **Custom Data**: User-provided datasets

### Performance Tuning

- **Thread Count**: Adjust for available CPU cores
- **Memory Settings**: Configure for available RAM
- **GPU Settings**: Optimize CUDA parameters for specific hardware

## Creating New Demo Files

### Demo File Structure

Each demo follows a consistent structure for both C++ and Python implementations:

#### C++ Demo Structure

1. **Include Headers**: Include necessary library headers and data loaders
2. **Configuration**: Set up variables (dataset size, dimensions, k, etc.)
3. **Data Loading**: Load or generate data and queries
4. **Algorithm Setup**: Create the search algorithm object
5. **Index Building**: Build the search index with the database
6. **Search Execution**: Perform the similarity search
7. **Results Display**: Print or process the search results
8. **Cleanup**: Free allocated memory

#### Python Demo Structure

1. **Imports**: Import required modules and libraries
2. **Configuration**: Set up parameters and variables
3. **Data Preparation**: Generate or load data using NumPy
4. **Algorithm Initialization**: Create the search algorithm instance
5. **Index Building**: Build the index with database points
6. **Search Execution**: Execute the search queries
7. **Results Processing**: Display or analyze the results

### Steps to Create a New Demo

#### For C++ Demos:

1. **Create the Demo File**

2. **Add Required Includes**

3. **Implement the Demo Logic**

4. **Update CMakeLists.txt**

#### For Python Demos:

1. **Create the Demo File**

2. **Add Required Imports**

3. **Implement the Demo Logic**
