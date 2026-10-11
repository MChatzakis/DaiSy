"""Coconut exact top-k across query workers, partial leaves and streaming inserts."""
import os
from pathlib import Path
import unittest

import numpy as np
import faiss
import daisy


@unittest.skipUnless(hasattr(daisy, 'Coconut'), 'Coconut is disabled')
class CoconutTopKTest(unittest.TestCase):
    def test_seed_boundaries_ties_and_partial_leaves(self):
        # Keys at either end of the directory; k exceeds the leaf size; ties and
        # best matches arriving in the streaming buffer must retain distinct IDs.
        data = np.repeat(np.arange(-20, 21, dtype=np.float32)[:, None], 64, axis=1)
        query = np.repeat(np.array([-100, -20, 0, 20, 100], dtype=np.float32)[:, None], 64, axis=1)
        inserted = np.concatenate((query, query))
        index = daisy.Coconut(daisy.DistanceType.L2_SQUARED)
        index.setLeafSize(7)
        index.setNumThreads(3)
        index.buildIndex(data)
        for add in (False, True):
            if add:
                index.insertBatch(inserted)
                data = np.concatenate((data, inserted))
            reference = faiss.IndexFlatL2(64)
            reference.add(data)
            for k in (1, 10, len(data), len(data) + 3):
                expected, _ = reference.search(query, k)
                ids, distances = index.searchIndex(query, k)
                count = min(k, len(data))
                np.testing.assert_allclose(distances, expected, rtol=2e-4, atol=2e-5)
                self.assertTrue(all(len(set(row[:count])) == count for row in ids))
                actual = np.sum((data[ids[:, :count].astype(np.int64)] - query[:, None]) ** 2, axis=2)
                np.testing.assert_allclose(distances[:, :count], actual, rtol=2e-4, atol=2e-5)
            config = daisy.SearchConfig()
            config.type = daisy.QueryType.RANGE
            config.r = 64.0
            ids, distances = index.searchIndex(query, config)
            for q, result_ids, result_dist in zip(query, ids, distances):
                exact = np.sum((data - q) ** 2, axis=1)
                self.assertEqual(set(result_ids), set(np.flatnonzero(exact <= config.r)))
                np.testing.assert_allclose(result_dist, np.sort(exact[exact <= config.r]))

    def test_empty_initial_index_and_invalid_configuration(self):
        index = daisy.Coconut(daisy.DistanceType.L2_SQUARED)
        index.buildIndex(np.empty((0, 64), dtype=np.float32))
        data = np.ones((5, 64), dtype=np.float32)
        index.insertBatch(data)
        ids, distances = index.searchIndex(data[:1], 5)
        self.assertEqual(len(set(ids[0])), 5)
        np.testing.assert_array_equal(distances, np.zeros((1, 5), dtype=np.float32))
        with self.assertRaisesRegex(RuntimeError, 'k > 0'):
            index.searchIndex(data[:1], 0)
        bad = daisy.Coconut(daisy.DistanceType.L2_SQUARED)
        bad.setLeafSize(0)
        with self.assertRaisesRegex(RuntimeError, 'positive leaf/PAA sizes'):
            bad.buildIndex(data)
        dtw = daisy.Coconut(daisy.DistanceType.DTW)
        dtw.buildIndex(data)
        with self.assertRaisesRegex(RuntimeError, 'only supports L2_SQUARED'):
            dtw.searchIndex(data[:1], 1)

    def test_build_does_not_retain_borrowed_input(self):
        rng = np.random.default_rng(891)
        data = rng.normal(size=(1500, 96)).astype(np.float32)
        query = data[[7, 503, 1499]].copy()
        reference = faiss.IndexFlatL2(96)
        reference.add(data)
        expected, _ = reference.search(query, 10)
        index = daisy.Coconut(daisy.DistanceType.L2_SQUARED)
        index.setNumThreads(3)
        index.buildIndex(data)
        # A successful build owns the raw series in leaves, even when it borrows
        # the input while writing them. Later changes to the array cannot affect it.
        data.fill(123.0)
        del data
        _, distances = index.searchIndex(query, 10)
        np.testing.assert_allclose(distances, expected, rtol=2e-4, atol=2e-5)

    def test_parallel_streaming_matches_faiss(self):
        rng = np.random.default_rng(194)
        data = rng.normal(size=(1600, 64)).astype(np.float32)
        # Queries include exact matches in both the built leaves and insert buffer.
        queries = np.concatenate((data[[0, 701, 900, 1499]], rng.normal(size=(12, 64)).astype(np.float32)))
        faiss.omp_set_num_threads(1)
        for leaf_size in (127, 1000):
            index = daisy.Coconut(daisy.DistanceType.L2_SQUARED)
            index.setLeafSize(leaf_size)
            index.buildIndex(data[:900])
            for stage in (900, 1200, 1600):
                if stage == 1200:
                    index.insertBatch(data[900:1200])
                elif stage == 1600:
                    index.insert(data[1200])
                    index.insertBatch(data[1201:1600])
                reference = faiss.IndexFlatL2(64)
                reference.add(data[:stage])
                for k in (1, 10, min(1000, stage)):
                    expected, _ = reference.search(queries, k)
                    for workers in (1, 4):
                        with self.subTest(leaf_size=leaf_size, stage=stage, k=k, workers=workers):
                            index.setNumThreads(workers)
                            ids, distances = index.searchIndex(queries, k)
                            np.testing.assert_allclose(distances, expected, rtol=2e-4, atol=2e-5)
                            self.assertTrue(all(len(set(row)) == k for row in ids))
                            actual = np.sum((data[ids.astype(np.int64)] - queries[:, None, :]) ** 2, axis=2)
                            np.testing.assert_allclose(distances, actual, rtol=2e-4, atol=2e-5)
                config = daisy.SearchConfig()
                config.type = daisy.QueryType.TOP_K
                config.k = 10
                _, distances = index.searchIndex(queries, config)
                expected, _ = reference.search(queries, 10)
                np.testing.assert_allclose(distances, expected, rtol=2e-4, atol=2e-5)

    def test_unreadable_leaf_raises(self):
        before = set(Path('/tmp').glob(f'daisy_coconut_{os.getpid()}_*'))
        index = daisy.Coconut(daisy.DistanceType.L2_SQUARED)
        data = np.zeros((20, 64), dtype=np.float32)
        index.buildIndex(data)
        index.setNumThreads(4)
        directories = set(Path('/tmp').glob(f'daisy_coconut_{os.getpid()}_*')) - before
        self.assertEqual(len(directories), 1)
        leaf = next(iter(directories)) / 'leaf.0'
        # A truncated file must never produce distances from an uninitialized buffer.
        leaf.write_bytes(b'')
        with self.assertRaisesRegex(RuntimeError, 'index leaf'):
            index.searchIndex(data[:4], 1)
        config = daisy.SearchConfig()
        config.type = daisy.QueryType.RANGE
        config.r = 1.0
        with self.assertRaisesRegex(RuntimeError, 'index leaf'):
            index.searchIndex(data[:4], config)


if __name__ == '__main__':
    unittest.main()
