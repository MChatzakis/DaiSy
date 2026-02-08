"""
daisy: High-performance similarity search library for time series data

This library provides multiple algorithms for nearest neighbor search on time series data:
- Brute Force search
- Lower Bound Brute Force search  
- MESSI - Advanced indexing algorithm for DTW searches
- Odyssey - MPI-based distributed search
- PARIS - Parallel indexing for disk-based datasets
- SING - CUDA-accelerated search
"""

__version__ = "1.0.0"
__author__ = ""

try:
    from .daisy import *
except ImportError:
    # If the compiled extension is not available, provide a helpful message
    import warnings
    warnings.warn(
        "daisy extension not found. "
        "The library may not have been compiled properly. "
        "Please check the installation instructions.",
        ImportWarning
    )
