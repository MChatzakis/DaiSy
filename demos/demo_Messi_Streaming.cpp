// MESSI streaming: build once, then update the live iSAX tree without rebuilding.

#include "../commons/dataloaders.hpp"
#include "../lib/daisy.hpp"

#include <cstdio>

int main()
{
    const daisy::idx_t dim = 96;
    const daisy::idx_t initial = 5000;
    const daisy::idx_t batch = 1000;
    const daisy::idx_t n_query = 5;
    const daisy::idx_t k = 5;

    float *stream = loadRandomData(initial + batch, dim, 100, true);
    float *query = loadRandomData(n_query, dim, 50, true);

    daisy::Messi search(daisy::DistanceType::L2_SQUARED);
    search.setIndexWorkers(2);
    search.setSearchWorkers(4);
    search.buildIndex(stream, initial, dim);

    daisy::idx_t *indices = new daisy::idx_t[n_query * k];
    float *distances = new float[n_query * k];

    // The index stays queryable between updates: search after every addition.
    auto queryAndReport = [&](const char *stage)
    {
        search.searchIndex(query, n_query, k, indices, distances);
        std::printf("%-16s MESSI contains %llu series. Query 0 kNN: ",
                    stage, search.getNDatabase());
        for (daisy::idx_t j = 0; j < k; ++j)
            std::printf("%llu(%.3f) ", indices[j], distances[j]);
        std::printf("\n");
    };

    queryAndReport("after build");

    search.insert(stream + initial * dim);
    queryAndReport("after insert");

    search.insertBatch(stream + (initial + 1) * dim, batch - 1);
    queryAndReport("after batch");

    delete[] stream;
    delete[] query;
    delete[] indices;
    delete[] distances;
    return 0;
}
