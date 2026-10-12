"""Exact MESSI queries across NumPy layouts, including header-strided fvecs views."""
import unittest
import numpy as np
import daisy


class MessiQueryLayoutTest(unittest.TestCase):
    def test_layouts_static_and_streaming(self):
        rng = np.random.default_rng(73)
        # Sparse positive descriptors expose the lowest adaptive SAX interval.
        database = rng.integers(0, 80, size=(1024, 128)).astype(np.float32)
        database[rng.random(database.shape) < 0.7] = 0
        queries = rng.integers(0, 80, size=(8, 128)).astype(np.float32)
        queries[rng.random(queries.shape) < 0.7] = 0
        header_rows = np.empty((8, 129), dtype=np.float32)
        header_rows[:, 0] = 128
        header_rows[:, 1:] = queries
        alternate_rows = np.empty((16, 128), dtype=np.float32)
        alternate_rows[::2] = queries
        layouts = {
            "dense": queries,
            "fvecs": header_rows[:, 1:],
            "row_slice": alternate_rows[::2],
            "fortran": np.asfortranarray(queries),
            "reversed_rows": queries[::-1],
            "reversed_columns": queries[:, ::-1],
        }
        for normalized in (False, True):
            for workers in (1, 4):
                index = daisy.Messi(daisy.DistanceType.L2_SQUARED)
                index.setNormalized(normalized)
                index.setIndexWorkers(2)
                index.setSearchWorkers(workers)
                initial = np.ascontiguousarray(database[:768])
                index.buildIndex(initial)
                for size in (768, 1024):
                    if size == 1024:
                        index.insertBatch(database[768:])
                    for name, query in layouts.items():
                        exact = ((database[:size].astype(np.float64)[None, :, :] -
                                  query.astype(np.float64)[:, None, :]) ** 2).sum(axis=2)
                        expected = np.sort(exact, axis=1)[:, :10]
                        config = daisy.SearchConfig()
                        config.type = daisy.QueryType.TOP_K
                        config.k = 10
                        for api in (10, config):
                            with self.subTest(normalized=normalized, workers=workers,
                                              size=size, layout=name, api=str(api)):
                                ids, distances = index.searchIndex(query, api)
                                ids = np.asarray(ids, dtype=np.int64)
                                distances = np.asarray(distances)
                                np.testing.assert_array_equal(distances, expected)
                                np.testing.assert_array_equal(
                                    distances, np.take_along_axis(exact, ids, axis=1))
                                for row in ids:
                                    self.assertEqual(len(set(row)), 10)
                        radius = float(np.median(expected[:, -1]))
                        config.type = daisy.QueryType.RANGE
                        config.r = radius
                        ids, distances = index.searchIndex(query, config)
                        for q, (row_ids, row_d) in enumerate(zip(ids, distances)):
                            self.assertEqual(set(row_ids), set(np.flatnonzero(exact[q] <= radius)))
                            np.testing.assert_array_equal(row_d, exact[q, row_ids])
                            self.assertEqual(len(row_ids), len(set(row_ids)))

    def test_invalid_query_shape_and_empty_index(self):
        index = daisy.Messi(daisy.DistanceType.L2_SQUARED)
        config = daisy.SearchConfig()
        config.k = 1
        for api in (1, config):
            with self.assertRaisesRegex(RuntimeError, "built"):
                index.searchIndex(np.zeros((1, 128), np.float32), api)
        database = np.arange(32 * 128, dtype=np.float32).reshape(32, 128)
        index.setNormalized(False)
        index.setIndexWorkers(1)
        index.buildIndex(database)
        for api in (1, config):
            with self.assertRaisesRegex(RuntimeError, "dimension"):
                index.searchIndex(np.zeros((2, 127), np.float32), api)
            with self.assertRaisesRegex(RuntimeError, "2D"):
                index.searchIndex(np.zeros(128, np.float32), api)
        config.k = 0
        with self.assertRaisesRegex(RuntimeError, "positive"):
            index.searchIndex(np.zeros((1, 128), np.float32), config)


if __name__ == "__main__":
    unittest.main()
