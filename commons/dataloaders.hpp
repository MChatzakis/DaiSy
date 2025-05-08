#ifndef DATALOADERS_HPP
#define DATALOADERS_HPP

float *loadBinData(const char *filename, unsigned long long n, unsigned long long dim);

float *z_normalize(const float *data, unsigned long long n, unsigned long long dim);

float *loadRandomData(unsigned long long n, unsigned long long dim, bool z_normalize = false, int seed = 0);

#endif // DATALOADERS_HPP