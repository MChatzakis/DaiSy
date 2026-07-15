"""
daisy: High-performance similarity search library for time series data

This library provides multiple algorithms for nearest neighbor search on time series data:
- BruteForceSearch  - exact brute-force baseline
- LbBruteforce      - brute-force with iSAX lower-bound pruning
- Messi             - in-memory iSAX index (L2 / DTW)
- Fresh             - FRESH in-memory iSAX index
- DumpyOS           - Dumpy-OS learned iSAX index
- Hercules          - hierarchical EAPCA+SAX index
- Sofa              - FFTW3 frequency-domain iSAX index (requires FFTW3)
- Sing              - CUDA-accelerated iSAX index
- ParIS             - parallel disk-based iSAX index
- Odyssey           - MPI distributed iSAX index

All algorithms support top-k search via searchIndex(query, k) and range
(distance-r) search via searchIndex(query, SearchConfig(type=QueryType.RANGE, r=r)).
"""

__version__ = "1.0.0"
__author__ = ""

try:
    from ._core import *
except ImportError:
    import warnings
    warnings.warn(
        "daisy._core extension not found. "
        "The library may not have been compiled properly. "
        "Please check the installation instructions.",
        ImportWarning
    )
