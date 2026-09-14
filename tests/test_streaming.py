import unittest

import numpy as np

from daisy import BruteForceSearch, DistanceType, LbBruteforce


class StreamingBindingsTest(unittest.TestCase):
    def test_bruteforce_and_lb_bruteforce(self):
        rng = np.random.default_rng(123)
        data = rng.normal(size=(8, 32)).astype(np.float32)

        for algorithm in (BruteForceSearch, LbBruteforce):
            with self.subTest(algorithm=algorithm.__name__):
                index = algorithm(DistanceType.L2_SQUARED)
                index.buildIndex(data[:3])
                index.insert(data[3])
                index.insertBatch(data[4:])

                indices, distances = index.searchIndex(data[7:8], 1)
                self.assertEqual(int(indices[0, 0]), 7)
                self.assertAlmostEqual(float(distances[0, 0]), 0.0, places=6)

    def test_dimension_validation(self):
        index = BruteForceSearch(DistanceType.L2_SQUARED)
        index.buildIndex(np.zeros((2, 32), dtype=np.float32))

        with self.assertRaises(RuntimeError):
            index.insert(np.zeros(31, dtype=np.float32))
        with self.assertRaises(RuntimeError):
            index.insertBatch(np.zeros((2, 31), dtype=np.float32))


if __name__ == "__main__":
    unittest.main()

