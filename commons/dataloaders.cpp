#include "dataloaders.hpp"

#include <iostream>
#include <cmath>

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

float *z_normalize(const float *data, unsigned long long n, unsigned long long dim)
{
    float *normalized_data = new float[n * dim];
    for (unsigned long long i = 0; i < n; i++)
    {
        float sum = 0.0f;
        for (unsigned long long j = 0; j < dim; j++)
        {
            sum += data[i * dim + j];
        }
        float mean = sum / dim;

        float sq_sum = 0.0f;
        for (unsigned long long j = 0; j < dim; j++)
        {
            sq_sum += (data[i * dim + j] - mean) * (data[i * dim + j] - mean);
        }
        float stddev = sqrt(sq_sum / dim);

        for (unsigned long long j = 0; j < dim; j++)
        {
            normalized_data[i * dim + j] = (data[i * dim + j] - mean) / stddev;
        }
    }
    return normalized_data;
}

float *loadRandomData(unsigned long long n, unsigned long long dim, bool z_norm = false, int seed = 0)
{
    if (seed != 0)
    {
        srand(seed);
    }
    else
    {
        srand(time(0));
    }

    float *data = new float[n * dim];
    for (unsigned long long i = 0; i < n; i++)
    {
        for (unsigned long long j = 0; j < dim; j++)
        {
            data[i * dim + j] = static_cast<float>(rand()) / RAND_MAX;
        }
    }

    if (z_norm)
    {
        float *normalized_data = z_normalize(data, n, dim);
        delete[] data;
        return normalized_data;
    }

    return data;
}