#pragma once

#include <iostream>
#include <variant>
#include <unordered_map>

namespace diNoLib
{
    using idx_t = unsigned long long;
    using ParameterValue = std::variant<int, float, bool, std::string>;

    class SimilaritySearchAlgorithm
    {
    protected:
        float *database;
        idx_t n_database;
        idx_t dim;

        virtual void setDefaultParameters() = 0;

    public:
        SimilaritySearchAlgorithm()
        {
            setDefaultParameters();
        }

        float *getDatabase() const { return database; }
        idx_t getNDatabase() const { return n_database; }
        idx_t getDim() const { return dim; }

        virtual void setIndexingParameters(const std::unordered_map<std::string, ParameterValue> &params) = 0;
        virtual void setQueryingParameters(const std::unordered_map<std::string, ParameterValue> &params) = 0;

        virtual void buildIndex(const float *database, const idx_t n_database, const idx_t dim) = 0;
        virtual void searchIndex(const float *query, const idx_t n_query, const idx_t k, idx_t *I, float *D) = 0;
        virtual ~SimilaritySearchAlgorithm() = default;
    };

}
