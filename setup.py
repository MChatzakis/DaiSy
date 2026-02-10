#!/usr/bin/env python

import os
import re
import sys
import shutil
from pathlib import Path
from setuptools import setup, Extension
from setuptools.command.build_py import build_py as _build_py
from setuptools.command.build_ext import build_ext as _build_ext

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
    from pybind11.setup_helpers import Pybind11Extension, build_ext as pybind_build_ext
    from pybind11 import get_cmake_dir
    import pybind11
    from setuptools import find_packages
    
    class build_ext(pybind_build_ext):
        """Custom build_ext that ensures daisy/__init__.py is in the lib directory"""
        def build_extensions(self):
            super().build_extensions()
            # After building extensions, copy __init__.py to the build lib directory
            src_file = Path("daisy") / "__init__.py"
            if src_file.exists():
                build_lib = Path(self.build_lib)
                dst_file = build_lib / "daisy" / "__init__.py"
                dst_file.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(str(src_file), str(dst_file))
    
    # Define the extension module using pybind11
    sources = [
        "pybinds/setup.cpp",
        # commons
        "commons/common.cpp",
        "commons/dataloaders.cpp",
        "commons/paramSetup.cpp",
        # lib - distance computers
        "lib/distance_computers/DistanceComputer.cpp",
        # lib - isax
        "lib/isax/SAX.cpp",
        "lib/isax/iSAXIndex.cpp",
        "lib/isax/iSAXPqueue.cpp",
        "lib/isax/iSAXSearch.cpp",
        # lib - utils
        "lib/utils/TimerManager.cpp",
        # lib - algos
        "lib/algos/Bruteforce.cpp",
        "lib/algos/LbBruteforce.cpp",
        "lib/algos/Messi.cpp",
        "lib/algos/ParIS.cpp",
        "lib/algos/Sing.cpp",
        # Odyssey - MPI-based distributed search (now required)
        "lib/algos/odyssey/Odyssey.cpp",
        "lib/algos/odyssey/bsf_sharing.cpp",
        "lib/algos/odyssey/indexing.cpp",
        "lib/algos/odyssey/replication.cpp",
        "lib/algos/odyssey/workstealing.cpp",
    ]
    
    # Get MPI include directories from mpi4py
    mpi_include_dirs = [
        "lib",
        "lib/algos", 
        "lib/distance_computers",
        "lib/isax",
        "lib/utils",
        "commons",
    ]
    
    # Add mpi4py include directories
    try:
        from mpi4py import get_include
        mpi_include_dirs.append(get_include())
    except Exception:
        pass
    
    # Add system MPI headers (from conda or system install)
    import os
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        conda_mpi_include = os.path.join(conda_prefix, "include")
        if conda_mpi_include not in mpi_include_dirs:
            mpi_include_dirs.append(conda_mpi_include)
    
    ext_modules = [
        Pybind11Extension(
            "daisy._core",
            sources,
            include_dirs=mpi_include_dirs,
            library_dirs=[os.path.join(conda_prefix, "lib")] if conda_prefix else [],
            libraries=["mpi"],  # Link against MPI library
            cxx_std=17,
            define_macros=[
                ("VERSION_INFO", '"' + __version__ + '"'),
                ("ODYSSEY_MPI", "1"),  # MPI always enabled
                ("SING_CUDA_ENABLED", "0"),
            ],
            extra_compile_args=["-fopenmp", "-mavx", "-march=native"],
            extra_link_args=["-fopenmp"],
        ),
    ]
    
    build_ext_class = build_ext
    
except ImportError:
    print("Warning: pybind11 not found. Extension will not be built.")
    print("Install pybind11 with: pip install pybind11")
    ext_modules = []
    build_ext_class = None

class build_py(_build_py):
    """Custom build_py to ensure daisy/__init__.py is included"""
    def run(self):
        super().run()
        # Ensure it's in the build lib
        src_file = Path("daisy") / "__init__.py"
        if src_file.exists():
            build_lib = Path(self.build_lib)
            dst_file = build_lib / "daisy" / "__init__.py"
            dst_file.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(str(src_file), str(dst_file))

setup(
    name="daisy-exact-search",
    version=__version__,
    author="",
    author_email="",  
    description="High-performance similarity search library for time series data",
    long_description=get_long_description(),
    long_description_content_type="text/markdown",
    url="https://github.com/MChatzakis/daisy",
    packages=find_packages(),
    include_package_data=True,
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext_class, "build_py": build_py} if build_ext_class else {"build_py": build_py},
    zip_safe=False,
    python_requires=">=3.10",
    install_requires=[
        "numpy>=2.2.6",
        "pybind11>=3.0.0",
        "mpi4py>=4.0.3",  # Required for Odyssey distributed search
    ],
    extras_require={
        "dev": [
            "build>=1.0.0",
            "cmake>=3.15",
            "pip>=24.0.0",
            "pytest>=7.0.0",
            "setuptools>=80.9.0",
            "twine>=4.0.0",
            "wheel>=0.45.1",
        ],
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
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    keywords="similarity-search time-series nearest-neighbor dtw cuda mpi",
    project_urls={
        "Bug Reports": "https://github.com/MChatzakis/daisy/issues",
        "Source": "https://github.com/MChatzakis/daisy",
    },
)
