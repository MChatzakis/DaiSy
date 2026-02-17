#include "dataloaders.hpp"

#include <cstdint>
#include <iostream>
#include <cmath>
#include <stdexcept>
#include <cstdio>
#include <cstring>

static float *z_normalize(const float *data, unsigned long long n, unsigned long long dim);

float *loadFbinData(const char *filename, size_t *dim_out, size_t *n_out, bool do_z_normalize)
{
    FILE *fp = fopen(filename, "rb");
    if (fp == nullptr)
    {
        throw std::runtime_error("(loadFbinData) Error opening file: " + std::string(filename));
    }

    int32_t d, n;
    if (fread(&d, sizeof(int32_t), 1, fp) != 1 || fread(&n, sizeof(int32_t), 1, fp) != 1)
    {
        fclose(fp);
        throw std::runtime_error("(loadFbinData) Error reading fbin header: " + std::string(filename));
    }
    if (d <= 0 || d > 1000000 || n <= 0)
    {
        fclose(fp);
        throw std::runtime_error("(loadFbinData) Invalid fbin header dim=" + std::to_string(d) + " n=" + std::to_string(n));
    }

    size_t dim = static_cast<size_t>(d);
    size_t count = static_cast<size_t>(n);

    float *data = new float[count * dim];
    size_t nr = fread(data, sizeof(float), count * dim, fp);
    fclose(fp);
    if (nr != count * dim)
    {
        delete[] data;
        throw std::runtime_error("(loadFbinData) Error reading fbin data: expected " + std::to_string(count * dim) + " floats, got " + std::to_string(nr));
    }

    *dim_out = dim;
    *n_out = count;

    if (!do_z_normalize)
    {
        return data;
    }

    float *normalized_data = z_normalize(data, count, dim);
    delete[] data;
    return normalized_data;
}

float *loadBinData(const char *filename, unsigned long long n, unsigned long long dim, bool do_z_normalize)
{
    FILE *fp = fopen(filename, "rb");
    if (fp == nullptr)
    {
        throw std::runtime_error("(loadBinData) Error opening file: " + std::string(filename));
    }

    float *data = new float[n * dim];
    size_t fread_out = fread(data, sizeof(float), n * (dim), fp);
    if (fread_out != n * dim)
    {
        delete[] data;
        fclose(fp);
        throw std::runtime_error("Error reading file: " + std::string(filename));
    }

    fclose(fp);

    if (!do_z_normalize)
    {
        std::cerr << "[loadBinData] DISCLAIMER: The library currently supports only searches on "
                     "normalized data and queries. It is therefore assumed that false is passed because the "
                     "loaded data are already z-normalized.\n";
        return data;
    }

    float *normalized_data = z_normalize(data, n, dim);
    delete[] data;
    return normalized_data;
}

static float *z_normalize(const float *data, unsigned long long n, unsigned long long dim)
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

float *loadRandomData(unsigned long long n, unsigned long long dim, int seed, bool do_z_normalize)
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

    if (!do_z_normalize)
    {
        std::cerr << "[loadRandomData] DISCLAIMER: The library currently supports only searches on "
                     "normalized data and queries. Future versions will support searches on raw data. "
                     "For the present release, the data will be normalized regardless of this parameter.\n";
    }

    float *normalized_data = z_normalize(data, n, dim);
    delete[] data;
    return normalized_data;
}