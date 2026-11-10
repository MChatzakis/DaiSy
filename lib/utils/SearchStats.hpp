#ifndef DAISY_SEARCH_STATS_HPP
#define DAISY_SEARCH_STATS_HPP

#include <cstdint>

namespace daisy
{
    // Work done by one exact top-k query. Algorithms that collect it return one
    // entry per query from SimilaritySearchAlgorithm::getLastSearchStats().
    // Currently: MESSI, FreSh, SOFA and single-node Odyssey, L2 squared only.
    //
    // The counts cover every phase of a query: the approximate seed (one leaf),
    // the serial refinement that runs when the seed found fewer than k series,
    // and the parallel exact phase. With several search threads the counts of
    // the same query vary from run to run, because the threads race on the
    // best-so-far distance that decides what gets pruned.
    // Not counted: the optional tight-bound pass of the refinement
    // (tight_bound = 1, off by default).
    //
    // The struct is kept trivial because worker structs that embed it live in
    // variable-length arrays; value-initialize it with SearchStats{}.
    struct SearchStats
    {
        uint64_t node_lb_calcs;   // lower bounds computed for tree nodes (refinement and exact phase)
        uint64_t leaves_enqueued; // leaves that passed node pruning in the exact phase and were queued
        uint64_t leaves_visited;  // leaf scans in all phases; a leaf scanned twice counts twice
        uint64_t series_lb_calcs; // lower bounds computed for single series inside scanned leaves
        uint64_t distance_calcs;  // real distance computations started (early-abandoned ones included)
        float seed_kth_dist;      // k-th best distance when the exact phase starts; FLT_MAX if the
                                  // seed and refinement found fewer than k series

        // Adds the counters of `o` (not seed_kth_dist).
        void addCounters(const SearchStats &o)
        {
            node_lb_calcs += o.node_lb_calcs;
            leaves_enqueued += o.leaves_enqueued;
            leaves_visited += o.leaves_visited;
            series_lb_calcs += o.series_lb_calcs;
            distance_calcs += o.distance_calcs;
        }
    };
}

#endif // DAISY_SEARCH_STATS_HPP
