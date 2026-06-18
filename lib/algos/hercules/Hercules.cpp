#include "../hhercules/Hercules.hpp"
#include "../hhercules/indexing.hpp"

#include <stdexcept>
#include <vector>

namespace daisy
{

Hercules::Hercules(DistanceType distance_type)
    : Hercules(distance_type, HerculesConfig{})
{
}
Hercules::Hercules(DistanceType distance_type, const HerculesConfig &config)
    : SimilaritySearchAlgorithm(distance_type), config_(config)
{
    this->leaf_size = config.leaf_size;
}

Hercules::~Hercules()
{
    destroy_tree(root_);
}

void Hercules::buildIndex(DataSource *data_source)
{
    int n = (int)data_source->getTotalRecords();
    int dim = (int)data_source->getDim();

    if (n == 0 || dim == 0)
        throw std::runtime_error("Hercules::buildIndex: empty data source");

    // need a raw pointer both for building and for writing leaf files to disk
    const float *raw = data_source->rawPointer();
    std::vector<float> tmp;
    if (!raw) {
        tmp.resize((size_t)n * dim);
        data_source->reset();
        for (int i = 0; i < n; i++)
            data_source->nextRecord(tmp.data() + (size_t)i * dim);
        raw = tmp.data();
    }

    destroy_tree(root_);
    root_ = nullptr;

    root_ = hercules_index_build(raw, n, dim, config_.leaf_size, 1, config_.num_build_threads);
    if (root_ == nullptr)
        throw std::runtime_error("Hercules::buildIndex: index build failed");

    if (!config_.index_dir.empty()) {
        if (!hercules_index_write(root_, raw, dim, config_.leaf_size, 1,
                                   config_.paa_segments, config_.sax_cardinality,
                                   config_.sax_bit_cardinality,
                                   config_.index_dir.c_str()))
            throw std::runtime_error("Hercules::buildIndex: index write failed");
    }

    this->n_database = (idx_t)n;
    this->dim = (idx_t)dim;
}
void Hercules::searchIndex(const float *query, idx_t n_query, idx_t k, idx_t *I, float *D)
{
    if (root_ == nullptr)
        throw std::runtime_error("Hercules::searchIndex: index not built");
    if (config_.index_dir.empty())
        throw std::runtime_error("Hercules::searchIndex: index_dir not set");

    for (idx_t q = 0; q < n_query; q++)
        hercules_knn_search(root_, query + q * this->dim, (int)this->dim, k,
                            I + q * k, D + q * k,
                            config_.index_dir.c_str(),
                            /*epsilon=*/0.0f,
                            config_.lmax,
                            config_.paa_segments,
                            (sax_type)config_.sax_bit_cardinality,
                            config_.sax_cardinality,
                            config_.eapca_th,
                            config_.sax_th,
                            (int)this->n_database,
                            config_.num_query_threads);
}

} 
