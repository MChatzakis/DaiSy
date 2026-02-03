#ifndef DATALOADERS_HPP
#define DATALOADERS_HPP

float *loadBinData(const char *filename, unsigned long long n, unsigned long long dim, bool do_z_normalize = true);

float *loadRandomData(unsigned long long n, unsigned long long dim, int seed = 0, bool do_z_normalize = true);

#endif // DATALOADERS_HPP