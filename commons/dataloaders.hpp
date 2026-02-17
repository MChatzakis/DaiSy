#ifndef DATALOADERS_HPP
#define DATALOADERS_HPP

#include <cstddef>

float *loadBinData(const char *filename, unsigned long long n, unsigned long long dim, bool do_z_normalize = true);

/** Load fvecs format: per-vector 4 bytes dim (int) then dim floats. Returns compact n*dim floats and sets *dim_out, *n_out. */
float *loadFvecsData(const char *filename, size_t *dim_out, size_t *n_out, bool do_z_normalize = true);

float *loadRandomData(unsigned long long n, unsigned long long dim, int seed = 0, bool do_z_normalize = true);

#endif 