"""Repeated leaf refinement must not repeat a seeded candidate in top-k."""
import unittest
import numpy as np
from daisy import DistanceType, Messi


class MessiTopkUniqueTest(unittest.TestCase):
    def test_seeded_candidates_remain_unique_after_refinement_and_inserts(self):
        # All records have zero PAA but different raw distances. Approximate and
        # exact phases revisit the same leaf; equal-valued rows still have distinct IDs.
        amplitudes = np.arange(1, 21, dtype=np.float32)
        amplitudes[1] = amplitudes[0]
        data = np.ascontiguousarray(amplitudes[:, None] * np.tile([1, -1], 16).astype(np.float32))
        queries = np.zeros((1, 32), dtype=np.float32)
        for normalized in (True, False):
            for metric in (DistanceType.L2_SQUARED, DistanceType.DTW):
                for threads in (1, 4):
                    index = Messi(metric)
                    index.setWarpingWindow(1)
                    index.setNormalized(normalized)
                    index.setIndexWorkers(threads)
                    index.setSearchWorkers(threads)
                    index.buildIndex(data[:12])
                    for end in (12, 20):
                        if end == 20:
                            index.insertBatch(data[12:])
                        expected = np.sort(np.sum(data[:end] ** 2, axis=1))
                        for k in (1, 3, 10):
                            with self.subTest(normalized=normalized, metric=metric, threads=threads, end=end, k=k):
                                ids, distances = index.searchIndex(queries, k)
                                self.assertEqual(len(np.unique(ids[0])), k)
                                np.testing.assert_allclose(distances[0], expected[:k])
                                np.testing.assert_allclose(distances[0], np.sum(data[ids[0].astype(np.int64)] ** 2, axis=1))
                    del index


if __name__ == '__main__':
    unittest.main()
