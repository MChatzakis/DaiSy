#pragma once

#include <iostream>
#include <variant>
#include <unordered_map>
#include <cfloat>

#include "../distance_computers/DistanceComputer.hpp"
#include "../isax/iSAXSearch.hpp"
#include "DataSource.hpp"

namespace daisy
{
    using idx_t = unsigned long long;

    class SimilaritySearchAlgorithm
    {
    protected:
        float *database = nullptr;
        idx_t n_database = 0;
        idx_t dim = 0;

        DistanceType distance_type;
        DistanceComputer *distance_computer = nullptr;

        int num_threads = 1;
        int warping_window = 10;
        int paa_segments = 16;
        int sax_cardinality = 8;
        int leaf_size = 2000;
        int min_leaf_size = 10;
        int initial_lbl_size = 2000;
        int flush_limit = 200000;
        int initial_fbl_size = 100;
        int total_loaded_leaves = 1;
        int tight_bound = 0;
        float minimum_distance = FLT_MAX;
        int min_checked_leaves = -1;

        isax_index_settings *index_settings = nullptr;
        isax_index *index = nullptr;
        sax_type **db_sax_representations = nullptr;

    public:
        SimilaritySearchAlgorithm(DistanceType distance_type)
        {
            this->distance_type = distance_type;
            this->distance_computer = new DistanceComputer(distance_type);
        }

        float *getDatabase() const { return database; }
        idx_t getNDatabase() const { return n_database; }
        idx_t getDim() const { return dim; }
        isax_index *getIndex() const { return index; } 

        virtual void buildIndex(DataSource *data_source) = 0;

        virtual void buildIndex(float *database, idx_t n_database, idx_t dim)
        {
            InMemoryDataSource data_source(database, n_database, dim);
            buildIndex(&data_source);
        }

        virtual void buildIndex(const std::string &filename, idx_t dim, idx_t n_database = 0)
        {
            FileDataSource data_source(filename.c_str(), dim, n_database);
            buildIndex(&data_source);
        }

    protected:

        bool validateSearchParams(const idx_t k, const idx_t n_query) const
        {
            if (k == 0)
            {
                std::cerr << "[Error] k must be greater than 0\n";
                return false;
            }
            if (k > n_database)
            {
                std::cerr << "[Error] k (" << k << ") cannot be greater than database size (" << n_database << ")\n";
                return false;
            }
            if (n_query == 0)
            {
                std::cerr << "[Error] n_query must be greater than 0\n";
                return false;
            }
            if (database == nullptr)
            {
                std::cerr << "[Error] Index must be built before searching\n";
                return false;
            }
            return true;
        }

    public:
        
        virtual void searchIndex(const float *query, const idx_t n_query, const idx_t k, idx_t *I, float *D) = 0;

        virtual void setNumThreads(int num_threads) {}

        virtual int getResultCompareRank() const { return 0; }

        virtual ~SimilaritySearchAlgorithm()
        {
            delete distance_computer;
        }
    };

}
