#include <gtest/gtest.h>
#include <isax/SAX.hpp>
#include <algorithm>
#include <array>
#include <vector>

namespace {
// Restore the process-wide table even if a fatal assertion exits the test.
struct ActiveTableGuard {
    const float *cuts = daisy::daisy_active_breakpoints;
    const float *max_cuts = daisy::daisy_active_breakpoints_max;
    ~ActiveTableGuard() { daisy::set_active_breakpoints(cuts, max_cuts); }
};
}

TEST(SaxBreakpointTest, ClassifiesShiftedRepeatedAndBoundaryCuts)
{
    ActiveTableGuard guard;
    for (int cardinality : {2, 4, 256}) {
        const int offset = (cardinality - 1) * (cardinality - 2) / 2;
        for (float shift : {-100.0f, 0.0f, 100.0f}) {
            // Poison the previous cardinality's last cut: it is not a lower
            // endpoint of this cardinality's first interval.
            std::vector<float> table(offset + cardinality - 1, 10000.0f);
            for (int i = 0; i < cardinality - 1; ++i)
                table[offset + i] = shift + i / 2;
            daisy::set_active_breakpoints(table.data(), nullptr);
            std::vector<float> probes{shift - 1, shift, shift + 0.25f,
                                      table.back(), table.back() + 1};
            for (int i = 0; i < cardinality - 1; ++i)
                probes.push_back(table[offset + i]);
            for (float value : probes) {
                int expected = 0;
                while (expected < cardinality - 1 && value > table[offset + expected])
                    ++expected;
                daisy::sax_type from_paa = 0, from_ts = 0;
                std::array<float, 8> series;
                series.fill(value);
                ASSERT_EQ(daisy::sax_from_paa(&value, &from_paa, 1, cardinality, 8), daisy::SUCCESS);
                ASSERT_EQ(daisy::sax_from_ts(series.data(), &from_ts, 8, 1, cardinality, 8), daisy::SUCCESS);
                EXPECT_EQ(from_paa, expected) << "cardinality=" << cardinality << " value=" << value;
                EXPECT_EQ(from_ts, expected) << "cardinality=" << cardinality << " value=" << value;
            }
        }
    }
}

TEST(SaxBreakpointTest, LowPositivePaaNeverOverestimatesRecordDistance)
{
    ActiveTableGuard guard;
    constexpr int cardinality = 256, segments = 16, values_per_segment = 8;
    constexpr int offset = (cardinality - 1) * (cardinality - 2) / 2;
    std::vector<float> table(offset + cardinality - 1, 200.0f);
    for (int i = 0; i < cardinality - 1; ++i)
        table[offset + i] = 2.0f + i / 4;
    daisy::set_active_breakpoints(table.data(), nullptr);
    std::array<float, segments> record_paa, query_paa;
    record_paa.fill(1.5f);
    query_paa.fill(3.0f);
    std::array<daisy::sax_type, segments> sax, bits;
    bits.fill(8);
    ASSERT_EQ(daisy::sax_from_paa(record_paa.data(), sax.data(), segments, cardinality, 8), daisy::SUCCESS);
    const float exact = segments * values_per_segment * (3.0f - 1.5f) * (3.0f - 1.5f);
    EXPECT_LE(daisy::minidist_paa_to_isax(query_paa.data(), sax.data(), bits.data(),
        8, cardinality, segments, -2000000, 2000000, values_per_segment, table.data()), exact);
    EXPECT_LE(daisy::minidist_paa_to_isax_rawa_SIMD(query_paa.data(), sax.data(), bits.data(),
        8, cardinality, segments, -2000000, 2000000, values_per_segment, table.data()), exact);
}
