import gc
import unittest
import weakref

import numpy as np

from daisy import BruteForceSearch, DistanceType, LbBruteforce, Messi


class StreamingBindingsTest(unittest.TestCase):
    def test_streaming_algorithms(self):
        rng = np.random.default_rng(123)
        data = rng.normal(size=(8, 32)).astype(np.float32)

        for algorithm in (BruteForceSearch, LbBruteforce, Messi):
            with self.subTest(algorithm=algorithm.__name__):
                index = algorithm(DistanceType.L2_SQUARED)
                if algorithm is Messi:
                    index.setIndexWorkers(1)
                    index.setSearchWorkers(1)
                index.buildIndex(data[:3])
                index.insert(data[3])
                index.insertBatch(data[4:])

                indices, distances = index.searchIndex(data[7:8], 1)
                self.assertEqual(int(indices[0, 0]), 7)
                self.assertAlmostEqual(float(distances[0, 0]), 0.0, places=6)

    def test_dimension_validation(self):
        for algorithm in (BruteForceSearch, LbBruteforce, Messi):
            with self.subTest(algorithm=algorithm.__name__):
                index = algorithm(DistanceType.L2_SQUARED)
                if algorithm is Messi:
                    index.setIndexWorkers(1)
                    index.setSearchWorkers(1)
                index.buildIndex(np.zeros((2, 32), dtype=np.float32))

                with self.assertRaises(RuntimeError):
                    index.insert(np.zeros(31, dtype=np.float32))
                with self.assertRaises(RuntimeError):
                    index.insertBatch(np.zeros((2, 31), dtype=np.float32))

    def test_messi_keeps_borrowed_build_array_alive(self):
        rng = np.random.default_rng(456)

        def build_from_temporary():
            data = rng.normal(size=(5, 32)).astype(np.float32)
            index = Messi(DistanceType.L2_SQUARED)
            index.setIndexWorkers(1)
            index.setSearchWorkers(1)
            borrowed = data[:3].copy()
            borrowed_ref = weakref.ref(borrowed)
            index.buildIndex(borrowed)
            return index, borrowed_ref, data[0].copy(), data[3].copy()

        index, borrowed_ref, initial, inserted = build_from_temporary()
        gc.collect()
        self.assertIsNotNone(borrowed_ref())
        initial_indices, initial_distances = index.searchIndex(initial.reshape(1, -1), 1)
        self.assertEqual(int(initial_indices[0, 0]), 0)
        self.assertAlmostEqual(float(initial_distances[0, 0]), 0.0, places=6)

        index.insert(inserted)
        indices, distances = index.searchIndex(inserted.reshape(1, -1), 1)
        self.assertEqual(int(indices[0, 0]), 3)
        self.assertAlmostEqual(float(distances[0, 0]), 0.0, places=6)

    def test_messi_build_rejects_temporary_numpy_conversions(self):
        index = Messi(DistanceType.L2_SQUARED)
        index.setIndexWorkers(1)
        index.setSearchWorkers(1)

        with self.assertRaises(RuntimeError):
            index.buildIndex(np.zeros((3, 32), dtype=np.float64))
        with self.assertRaises(RuntimeError):
            index.buildIndex(np.zeros((3, 64), dtype=np.float32)[:, ::2])


if __name__ == "__main__":
    unittest.main()
