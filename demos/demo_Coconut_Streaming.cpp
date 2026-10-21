// COCONUT — streaming: build on an initial batch, then insert new series into the live
// index as they arrive and query at any point (no rebuild). COCONUT's distinguishing feature.

#include "../commons/dataloaders.hpp"
#include "../lib/daisy.hpp"

int main()
{
    unsigned long long dim = 96;
    daisy::idx_t initial = 50000;   // series available at build time
    daisy::idx_t batch = 25000;     // series that arrive later, per streaming step
    int steps = 3;
    unsigned long long n_query = 5;
    daisy::idx_t k = 5;

    // Whole stream in one buffer; we feed it in pieces to simulate arrival over time.
    daisy::idx_t total = initial + (daisy::idx_t)steps * batch;
    float *stream = loadRandomData(total, dim, 100, true);
    float *query = loadRandomData(n_query, dim, 50, true);

    daisy::Coconut coconut(daisy::DistanceType::L2_SQUARED);
    coconut.setBufferCapacity(10000);  // auto flush and background binary disk merges
    coconut.setNumThreads(2);
    coconut.setQueryBatchSize(32);       // shared reads; exact per-query SAX bounds/heaps

    // 1. Build on the initial batch.
    coconut.buildIndex(stream, initial, dim);
    printf("Built on %llu series.\n", initial);

    daisy::idx_t *I = new daisy::idx_t[n_query * k];
    float *D = new float[n_query * k];

    auto run_query = [&](const char *when)
    {
        coconut.searchIndex(query, n_query, k, I, D);
        printf("[%s] query 0 kNN: ", when);
        for (daisy::idx_t j = 0; j < k; j++)
            printf("%llu(%.1f) ", I[j], D[j]);
        printf("\n");
    };
    run_query("after build");

    // 2. Stream the remaining series in batches, querying after each.
    daisy::idx_t seen = initial;
    for (int s = 0; s < steps; s++)
    {
        coconut.insertBatch(stream + (size_t)seen * dim, batch);
        coconut.waitForMerges(); // include pending compaction in completed-insert timing
        seen += batch;
        printf("Streamed a batch; index now holds %llu series.\n", seen);
        printf("Disk records: %llu; buffered: %llu; merges: %llu\n",
               coconut.getDiskRecords(), coconut.getBufferedRecords(), coconut.getMergeCount());
        run_query("after insert");
        coconut.searchIndexRecent(query, n_query, 1, batch, I, D);
        printf("Most recent batch, query 0 top-1: %llu (%.1f)\n", I[0], D[0]);
    }

    delete[] stream;
    delete[] query;
    delete[] I;
    delete[] D;

    return 0;
}
