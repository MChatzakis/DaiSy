# include "dataloaders.hpp"

#include <iostream>

float *loadBinData(const char *filename, unsigned long long n, unsigned long long dim)
{
    FILE *fp = fopen(filename, "rb");
    if (fp == nullptr)
    {
        std::cerr << "Error opening file: " << filename << std::endl;
        exit(EXIT_FAILURE);
    }

    float *data = new float[n * dim];
    size_t fread_out = fread(data, sizeof(float), n * (dim), fp);
    if (fread_out != n * dim)
    {
        std::cerr << "Error reading file: " << filename << std::endl;
        delete[] data;
        fclose(fp);
        exit(EXIT_FAILURE);
    }

    fclose(fp);
    return data;
}