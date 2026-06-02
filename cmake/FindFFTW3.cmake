# FindFFTW3.cmake - Find FFTW3 library
# Sets: FFTW3_FOUND, FFTW3::fftw3f (single precision imported target)

find_path(FFTW3_INCLUDE_DIR
    NAMES fftw3.h
    PATHS /usr/include /usr/local/include /opt/local/include
)

find_library(FFTW3f_LIBRARY
    NAMES fftw3f libfftw3f
    PATHS /usr/lib /usr/local/lib /usr/lib/x86_64-linux-gnu
          /usr/lib64 /opt/local/lib
)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(FFTW3
    REQUIRED_VARS FFTW3f_LIBRARY FFTW3_INCLUDE_DIR
)

if(FFTW3_FOUND AND NOT TARGET FFTW3::fftw3f)
    add_library(FFTW3::fftw3f UNKNOWN IMPORTED)
    set_target_properties(FFTW3::fftw3f PROPERTIES
        IMPORTED_LOCATION "${FFTW3f_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${FFTW3_INCLUDE_DIR}"
    )
endif()

mark_as_advanced(FFTW3_INCLUDE_DIR FFTW3f_LIBRARY)
