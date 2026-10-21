"""Regression: early abandonment must not accept a partial sum equal to r."""
import unittest
import numpy as np
from daisy import DistanceType, Messi, QueryType, SearchConfig


class RangeBoundaryTest(unittest.TestCase):
    def test_partial_sum_and_zero_radius(self):
        data = np.zeros((3, 32), dtype=np.float32)
        # Paired signs give identical PAA summaries, so refinement is exercised.
        data[1, :16] = np.tile([1, -1], 8)  # partial distance 8, full distance 16
        data[2, 8:16] = np.tile([1, -1], 4)  # full distance exactly 8
        query = np.zeros((1, 32), dtype=np.float32)
        for normalized in (True, False):
            index = Messi(DistanceType.L2_SQUARED)
            index.setIndexWorkers(1)
            index.setSearchWorkers(1)
            index.setNormalized(normalized)
            index.buildIndex(data)
            for radius, expected in ((0.0, {0}), (8.0, {0, 2})):
                with self.subTest(normalized=normalized, radius=radius):
                    cfg = SearchConfig()
                    cfg.type = QueryType.RANGE
                    cfg.r = radius
                    ids, distances = index.searchIndex(query, cfg)
                    self.assertEqual(set(ids[0]), expected)
                    for i, d in zip(ids[0], distances[0]):
                        self.assertEqual(d, float(np.sum(data[i] ** 2)))
            del index


if __name__ == '__main__':
    unittest.main()
