#ifndef DATALOADERS_HPP
#define DATALOADERS_HPP

float *loadBinData(const char *filename, unsigned long long n, unsigned long long dim);

float *loadRandomData(unsigned long long n, unsigned long long dim, int seed = 0);

#endif // DATALOADERS_HPP