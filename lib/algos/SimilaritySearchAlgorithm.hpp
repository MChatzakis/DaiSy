#pragma once

#include <iostream>
#include <variant>
#include <unordered_map>
#include <cfloat>

#include "../distance_computers/DistanceComputer.hpp"
#include "../isax/iSAXSearch.hpp"

namespace diNoLib
{
    using idx_t = unsigned long long;
    //using ParameterValue = std::variant<int, float, bool, std::string>;

    class SimilaritySearchAlgorithm
    {
    protected:
        float *database = nullptr;
        idx_t n_database = 0;
        idx_t dim = 0;

        DistanceType distance_type;
        DistanceComputer *distance_computer = nullptr;
        
        int num_threads = 1;
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

        virtual void buildIndex(const float *database, const idx_t n_database, const idx_t dim) = 0;
        virtual void searchIndex(const float *query, const idx_t n_query, const idx_t k, idx_t *I, float *D) = 0;
        /**
         * @param query a pointor to an array of float
         * @param n_query number of query vectors
         * @param k number of nearest neighbors
         * @param I output indices of nearest neighbors
         * @param D output distances of nearest neighbors
        */ 
        
        virtual void setNumThreads(int num_threads) {}

        virtual ~SimilaritySearchAlgorithm()
        {
            delete distance_computer;
        }
    };

}
