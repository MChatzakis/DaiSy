#!/usr/bin/env python

import os
import re
import sys
from pathlib import Path
from setuptools import setup, Extension

# Version
__version__ = "1.0.0"

def get_long_description():
    """Get long description from README"""
    readme_path = Path(__file__).parent / "README.md"
    if readme_path.exists():
        return readme_path.read_text(encoding="utf-8")
    return "High-performance similarity search library for time series data"

# Try to import pybind11
try:
    from pybind11.setup_helpers import Pybind11Extension, build_ext
    from pybind11 import get_cmake_dir
    import pybind11
    
    # Define the extension module using pybind11
    ext_modules = [
        Pybind11Extension(
            "diNoSimilaritySearch",
            [
                "pybinds/setup.cpp",
            ],
            include_dirs=[
                "lib",
                "lib/algos", 
                "lib/distance_computers",
                "lib/utils",
                "commons",
            ],
            cxx_std=17,
            define_macros=[("VERSION_INFO", '"' + __version__ + '"')],
        ),
    ]
    
    build_ext_class = build_ext
    
except ImportError:
    print("Warning: pybind11 not found. Extension will not be built.")
    print("Install pybind11 with: pip install pybind11")
    ext_modules = []
    build_ext_class = None

setup(
    name="diNo-lib",
    version=__version__,
    author="",
    author_email="",  
    description="High-performance similarity search library for time series data",
    long_description=get_long_description(),
    long_description_content_type="text/markdown",
    url="https://github.com/MChatzakis/diNo-lib",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext_class} if build_ext_class else {},
    zip_safe=False,
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.19.0",
        "pybind11>=2.10.0",
    ],
    extras_require={
        "mpi": ["mpi4py>=4.0.0"],
        "dev": ["pytest", "cmake"],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: POSIX :: Linux",
        "Operating System :: MacOS :: MacOS X", 
        "Operating System :: Microsoft :: Windows",
        "Programming Language :: C++",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    keywords="similarity-search time-series nearest-neighbor dtw cuda mpi",
    project_urls={
        "Bug Reports": "https://github.com/MChatzakis/diNo-lib/issues",
        "Source": "https://github.com/MChatzakis/diNo-lib",
    },
)
