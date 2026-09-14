// Bruteforce streaming: build once, then append new series without rebuilding.

#include "../commons/dataloaders.hpp"
#include "../lib/daisy.hpp"

#include <cstdio>

int main()
{
    const daisy::idx_t dim = 96;
    const daisy::idx_t initial = 50000;
    const daisy::idx_t batch = 25000;
    const daisy::idx_t n_query = 5;
    const daisy::idx_t k = 5;

    float *stream = loadRandomData(initial + batch, dim, 100, true);
    float *query = loadRandomData(n_query, dim, 50, true);

    daisy::BruteForceSearch search(daisy::DistanceType::L2_SQUARED);
    search.buildIndex(stream, initial, dim);
    search.insert(stream + initial * dim);
    search.insertBatch(stream + (initial + 1) * dim, batch - 1);

    daisy::idx_t *indices = new daisy::idx_t[n_query * k];
    float *distances = new float[n_query * k];
    search.searchIndex(query, n_query, k, indices, distances);

    std::printf("Bruteforce now contains %llu series. Query 0 kNN: ",
                search.getNDatabase());
    for (daisy::idx_t j = 0; j < k; ++j)
        std::printf("%llu(%.3f) ", indices[j], distances[j]);
    std::printf("\n");

    delete[] stream;
    delete[] query;
    delete[] indices;
    delete[] distances;
    return 0;
}

