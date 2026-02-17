#include "dataloaders.hpp"

#include <cstdint>
#include <iostream>
#include <cmath>
#include <stdexcept>
#include <cstdio>
#include <cstring>
#include <sys/stat.h>

static float *z_normalize(const float *data, unsigned long long n, unsigned long long dim);

float *loadFvecsData(const char *filename, size_t *dim_out, size_t *n_out, bool do_z_normalize)
{
    /* Same logic as fvecs_read from VectorDataLoader: fopen "r", read int d, fstat, n = sz/((d+1)*4), read n*(d+1) floats, memmove to compact. */
    FILE *f = fopen(filename, "r");
    if (!f)
    {
        throw std::runtime_error("(loadFvecsData) could not open " + std::string(filename));
    }

    int d;
    if (fread(&d, 1, sizeof(int), f) != sizeof(int))
    {
        fclose(f);
        throw std::runtime_error("(loadFvecsData) could not read dimension: " + std::string(filename));
    }
    if (d <= 0 || d > 1000000)
    {
        fclose(f);
        throw std::runtime_error("(loadFvecsData) unreasonable dimension " + std::to_string(d) + ": " + std::string(filename));
    }

    fseek(f, 0, SEEK_SET);
    struct stat st;
    if (fstat(fileno(f), &st) != 0)
    {
        fclose(f);
        throw std::runtime_error("(loadFvecsData) fstat failed: " + std::string(filename));
    }
    size_t sz = static_cast<size_t>(st.st_size);
    size_t block = (static_cast<size_t>(d) + 1) * 4;
    if (sz % block != 0)
    {
        fclose(f);
        throw std::runtime_error("(loadFvecsData) weird file size " + std::to_string(sz) + " for dim " + std::to_string(d));
    }
    size_t n = sz / block;

    /* Use unsigned long long to avoid overflow for 100M * (d+1); then check it fits in size_t for allocation */
    unsigned long long total_ull = static_cast<unsigned long long>(n) * static_cast<unsigned long long>(d + 1);
    if (total_ull > static_cast<unsigned long long>(SIZE_MAX))
    {
        fclose(f);
        throw std::runtime_error("(loadFvecsData) file too large for allocation: n=" + std::to_string(n) + " d+1=" + std::to_string(d + 1));
    }
    size_t total = static_cast<size_t>(total_ull);

    float *x = new float[total];
    size_t nr = 0;
    while (nr < total)
    {
        size_t got = fread(x + nr, sizeof(float), total - nr, f);
        if (got == 0)
            break;
        nr += got;
    }
    fclose(f);
    if (nr != total)
    {
        delete[] x;
        throw std::runtime_error("(loadFvecsData) could not read whole file: read " + std::to_string(nr) + " of " + std::to_string(total) + ": " + std::string(filename));
    }

    for (size_t i = 0; i < n; i++)
    {
        memmove(x + i * d, x + 1 + i * (d + 1), d * sizeof(*x));
    }

    unsigned long long out_size_ull = static_cast<unsigned long long>(n) * static_cast<unsigned long long>(d);
    size_t out_size = (out_size_ull <= static_cast<unsigned long long>(SIZE_MAX)) ? static_cast<size_t>(out_size_ull) : 0;
    if (out_size == 0 && out_size_ull != 0)
    {
        delete[] x;
        throw std::runtime_error("(loadFvecsData) result too large for size_t");
    }
    float *out = new float[out_size];
    std::memcpy(out, x, out_size * sizeof(float));
    delete[] x;

    *dim_out = static_cast<size_t>(d);
    *n_out = n;

    if (!do_z_normalize)
    {
        return out;
    }
    float *normalized = z_normalize(out, n, static_cast<unsigned long long>(d));
    delete[] out;
    return normalized;
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