#pragma once

#include <iostream>
#include <variant>
#include <unordered_map>

#include "../distance_computers/DistanceComputer.hpp"

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

        virtual ~SimilaritySearchAlgorithm()
        {
            delete distance_computer;
        }
    };

}
