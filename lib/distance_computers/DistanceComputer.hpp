#ifndef DISTANCECOMPUTER_HPP
#define DISTANCECOMPUTER_HPP

#include "immintrin.h"
#include <cmath>
#include <unordered_map>
#include <iostream>

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

        float compute_dist(float *t, float *s, int dim, float bound);
    };

} // namespace diNoLib


#endif // DISTANCECOMPUTER_HPP