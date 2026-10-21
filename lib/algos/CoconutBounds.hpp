#ifndef DAISY_COCONUT_BOUNDS_HPP
#define DAISY_COCONUT_BOUNDS_HPP
#include "../isax/SAX.hpp"
#include <cmath>
#include <vector>

namespace daisy { namespace coconut_detail {
// Full-cardinality SAX interval distances for a single query. Double table
// entries preserve the reference helper's pow(float_delta, 2) contributions;
// the float accumulator keeps its original per-segment rounding/order.
class QueryBounds
{
    int segments_ = 0, alphabet_ = 0;
    float ratio_ = 1;
    std::vector<double> contribution_;
public:
    void reset(const float *paa, int segments, int bits, float ratio)
    {
        segments_ = segments; alphabet_ = 1 << bits; ratio_ = ratio;
        const int offset = (alphabet_ - 1) * (alphabet_ - 2) / 2;
        contribution_.resize((size_t)segments * alphabet_);
        for (int d = 0; d < segments; ++d)
            for (int symbol = 0; symbol < alphabet_; ++symbol)
            {
                float low = symbol ? sax_breakpoints[offset + symbol - 1] : MINVAL;
                float high = symbol == alphabet_ - 1 ? MAXVAL : sax_breakpoints[offset + symbol];
                float delta = low > paa[d] ? low - paa[d] : high < paa[d] ? high - paa[d] : 0;
                contribution_[(size_t)d * alphabet_ + symbol] = std::pow(delta, 2);
            }
    }
    float operator()(const sax_type *word, float limit) const
    {
        float distance = 0;
        for (int d = 0; d < segments_; ++d)
        {
            distance += contribution_[(size_t)d * alphabet_ + word[d]];
            // All contributions are nonnegative. A partial lower bound above
            // the current bound safely prunes without evaluating later segments.
            if ((d & 3) == 3 && ratio_ * distance > limit) return ratio_ * distance;
        }
        return ratio_ * distance;
    }
    size_t bytes() const { return contribution_.capacity() * sizeof(double); }
};
}}
#endif
