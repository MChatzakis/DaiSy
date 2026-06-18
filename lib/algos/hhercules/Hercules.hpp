#pragma once

#include "../SimilaritySearchAlgorithm.hpp"

#include <stdexcept>
#include <string>

namespace daisy
{

// forward-declaration —> keeps indexing internals out of this public header
struct HerculesNode;

struct HerculesConfig
{
    int leaf_size = 2000;
    int lmax = 80;
    float eapca_th = 0.25f;
    float sax_th = 0.50f;
    int paa_segments = 16;
    int sax_bit_cardinality = 3;
    int sax_cardinality = 8;
    int num_build_threads = 1;
    int num_query_threads = 1;
    int flush_threshold = 200000;
    int insert_buffer_size = 1000;
    int flush_buffer_size = 10000;
    std::string index_dir = "";
};

class Hercules : public SimilaritySearchAlgorithm
{
public:
    Hercules(DistanceType distance_type);
    Hercules(DistanceType distance_type, const HerculesConfig &config);

    using SimilaritySearchAlgorithm::buildIndex;

    void buildIndex(DataSource *data_source) override;

    void searchIndex(const float *query, idx_t n_query, idx_t k,
                     idx_t *I, float *D) override;

    ~Hercules() override;

private:
    HerculesConfig config_;
    HerculesNode  *root_ = nullptr;
};

} 
