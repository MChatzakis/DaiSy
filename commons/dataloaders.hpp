#ifndef DATALOADERS_HPP
#define DATALOADERS_HPP

#include <cstddef>

float *loadBinData(const char *filename, unsigned long long n, unsigned long long dim, bool do_z_normalize = true);

/** Load fbin format: 4 bytes dim (int32), 4 bytes n (int32), then n*dim floats. Returns data and sets *dim_out, *n_out. */
float *loadFbinData(const char *filename, size_t *dim_out, size_t *n_out, bool do_z_normalize = true);

float *loadRandomData(unsigned long long n, unsigned long long dim, int seed = 0, bool do_z_normalize = true);

#endif 