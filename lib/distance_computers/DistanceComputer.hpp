#ifndef DISTANCECOMPUTER_HPP
#define DISTANCECOMPUTER_HPP

#include "immintrin.h"
#include <cmath>
#include <unordered_map>
#include <iostream>

#include "../lib/isax/iSAXTypes.hpp"
#include "../lib/isax/SAX.hpp"

namespace diNoLib
{
    enum class DistanceType
    {
        L2_SQUARED = 0,
        // More to be implemented
        DTW = 1
    };
}

namespace std {
    template<>
    struct hash<diNoLib::DistanceType>
    {
        std::size_t operator()(const diNoLib::DistanceType& dt) const noexcept
        {
            return std::hash<int>()(static_cast<int>(dt));
        }
    };
}

namespace diNoLib
{
    class DistanceComputer
    {
    private:
        DistanceType distance_type;

        std::unordered_map<DistanceType, float (DistanceComputer::*)(float *, float *, int, float)> distance_map;
        
        void init_distance_map();
        float l2_dist(float *t, float *s, int dim, float bound);
        float l2_dist_SIMD(float *t, float *s, int dim, float bound);
        float l2_dist_naive(float *t, float *s, int dim, float bound);

    public:
        DistanceComputer(DistanceType distance_type)
        {
            init_distance_map();

            this->distance_type = distance_type;

            if (distance_map.find(distance_type) == distance_map.end())
            {
                throw std::invalid_argument("Distance type not supported");
            }
        }

        ~DistanceComputer()
        {
            // Destructor
        }

        float compute_dist(float *t, 
                            float *s, 
                            int dim, 
                            float bound);
        
        float compute_minidist_SIMD(const ts_type *q_paa,
                            const sax_type *db_sax,
                            const int *max_sax_cardinalities,
                            int sax_bit_cardinality,
                            int sax_alphabet_cardinality,
                            int paa_segments,
                            float minval,
                            float maxval,
                            bool mindist_sqrt);

        void compute_paa_from_ts(const float *ts,
                                ts_type *paa,
                                int paa_segments,
                                int ts_values_per_segment);

        bool compute_sax_from_ts(const float *ts,
                                sax_type *sax,
                                int ts_values_per_paa_segment,
                                int paa_segments,
                                int sax_alphabet_cardinality,
                                int sax_bit_cardinality);     
                                
        float compute_dist_SIMD(float *t, 
                                float *s, 
                                int dim, 
                                float bound);                                
    };

} // namespace diNoLib


#endif // DISTANCECOMPUTER_HPP