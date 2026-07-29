#!/usr/bin/env python

import os
import re
import sys
import shutil
import platform
from pathlib import Path
from setuptools import setup, Extension, find_packages
from setuptools.command.build_py import build_py as _build_py
from setuptools.command.build_ext import build_ext as _build_ext

# Version
__version__ = "1.0.9"

# Detect platform
IS_MACOS = sys.platform == "darwin"
IS_WINDOWS = sys.platform == "win32"
IS_LINUX = sys.platform.startswith("linux")

def check_mpi_available():
    """Check if MPI development libraries are available on the system (mpi.h for C++ compilation).
    Note: mpicc or mpi4py alone might not be sufficient."""
    import subprocess
    #import os
    
    # Check if mpi.h exists
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        mpi_header = os.path.join(conda_prefix, "include", "mpi.h")
        if os.path.exists(mpi_header):
            return True
    
    # Check common system paths
    for path in ["/usr/include", "/usr/local/include", "/opt/local/include", "/sw/include"]:
        if os.path.exists(os.path.join(path, "mpi.h")):
            return True
    
    # Try to run mpicc
    try:
        result = subprocess.run(["mpicc", "--version"], capture_output=True, timeout=5)
        if result.returncode == 0:
            return True
    except (FileNotFoundError, subprocess.TimeoutExpired, PermissionError):
        # Any problem running mpicc is treated as "MPI not available"
        pass
    
    # Try to import mpi4py  
    #try:
    #    import mpi4py
    #    return True
    #except ImportError:
    #    pass
    
    return False

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
    # Odyssey (distributed search with MPI) is enabled ONLY if explicitly requested
    # via the ENABLE_MPI environment variable. This avoids probing MPI on systems
    # where the user did not ask for it.
    ENABLE_MPI = os.environ.get("ENABLE_MPI", "0").lower() in ("1", "true", "yes")

    if ENABLE_MPI:
        if IS_MACOS or IS_WINDOWS:
            raise RuntimeError(
                "[DaiSy] ENABLE_MPI is set but Odyssey/MPI is not supported on this platform."
            )
        if not check_mpi_available():
            raise RuntimeError(
                "[DaiSy] ENABLE_MPI is set but MPI development libraries are not available.\n"
                "[DaiSy] Please install an MPI implementation (e.g., OpenMPI/MPICH) or unset ENABLE_MPI."
            )
        ODYSSEY_ENABLED = True
    else:
        ODYSSEY_ENABLED = False
    
    # Inform user about build configuration
    if ODYSSEY_ENABLED:
        print("[DaiSy] Building with Odyssey (distributed search) support - MPI enabled (ENABLE_MPI=1)")
    else:
        print("[DaiSy] Building without Odyssey support (ENABLE_MPI is not set or false)")
    
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
    ]
    
    # Add Odyssey sources only if MPI is available
    if ODYSSEY_ENABLED:
        odyssey_sources = [
            "lib/algos/odyssey/Odyssey.cpp",
            "lib/algos/odyssey/bsf_sharing.cpp",
            "lib/algos/odyssey/indexing.cpp",
            "lib/algos/odyssey/replication.cpp",
            "lib/algos/odyssey/workstealing.cpp",
        ]
        sources.extend(odyssey_sources)
    
    # Get MPI include directories from mpi4py (only if available)
    mpi_include_dirs = [
        "lib",
        "lib/algos", 
        "lib/distance_computers",
        "lib/isax",
        "lib/utils",
        "commons",
    ]
    
    library_dirs = []
    libraries = []
    
    if ODYSSEY_ENABLED:
        # Add mpi4py include directories
        try:
            from mpi4py import get_include
            mpi_include_dirs.append(get_include())
        except Exception:
            pass
        
        # Add system MPI headers (from conda or system install)
        conda_prefix = os.environ.get("CONDA_PREFIX")
        if conda_prefix:
            conda_mpi_include = os.path.join(conda_prefix, "include")
            if conda_mpi_include not in mpi_include_dirs:
                mpi_include_dirs.append(conda_mpi_include)
            library_dirs.append(os.path.join(conda_prefix, "lib"))
        
        libraries.append("mpi")
    
    # Determine compiler flags based on platform
    compile_args = ["-std=c++17"]
    link_args = []
    
    # OpenMP support (not available on standard macOS clang)
    if not IS_MACOS:
        compile_args.append("-fopenmp")
        link_args.append("-fopenmp")
    
    # SIMD optimizations (not reliable on ARM64 Macs)
    #
    # -march=native tunes for the machine doing the build, which is what you
    # want when compiling from source but not when building a wheel for other
    # people: SIMD.hpp picks its code path at compile time, so a wheel built on
    # a runner with a wider instruction set dies with SIGILL elsewhere. Set
    # DAISY_PORTABLE_BUILD=1 when producing a redistributable wheel; -mavx
    # stays, so DAISY_SIMD_AVAILABLE is still 1.
    PORTABLE_BUILD = os.environ.get("DAISY_PORTABLE_BUILD", "0").lower() in ("1", "true", "yes")

    if not (IS_MACOS and platform.machine() in ("arm64", "aarch64")):
        compile_args.append("-mavx")
        if PORTABLE_BUILD:
            # The guarded code needs AVX2, not just AVX, so a portable wheel has
            # to name AVX2 explicitly once -march=native is gone. That puts the
            # floor at Haswell (2013).
            print("[DaiSy] Portable build requested - targeting AVX2 instead of -march=native")
            compile_args.append("-mavx2")
        else:
            compile_args.append("-march=native")
    
    # Get the project root directory as absolute path
    project_root = str(Path(__file__).parent.absolute())
    
    define_macros = [
        ("VERSION_INFO", '"' + __version__ + '"'),
        ("BUILD_ODYSSEY", "1" if ODYSSEY_ENABLED else "0"),
        ("ODYSSEY_MPI", "1" if ODYSSEY_ENABLED else "0"),  # Used by pybinds and Odyssey headers
        ("SING_CUDA_ENABLED", "0"),
        ("PROJECT_ROOT_DIR", f'"{project_root}"'),
    ]
    
    ext_modules = [
        Pybind11Extension(
            "daisy._core",
            sources,
            include_dirs=mpi_include_dirs,
            library_dirs=library_dirs,
            libraries=libraries,
            cxx_std=17,
            define_macros=define_macros,
            extra_compile_args=compile_args,
            extra_link_args=link_args,
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
    ],
    extras_require={
        "mpi": ["mpi4py>=4.0.3"],  # Optional: for Odyssey distributed search
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