"""Top-k with k larger than the indexed series must raise ValueError, never pad."""
import os
import tempfile
import unittest

import numpy as np
import daisy
from daisy import DistanceType

N, DIM = 16, 64


class TopKGreaterThanNTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(3)
        cls.data = np.ascontiguousarray(rng.standard_normal((N, DIM)).astype(np.float32))
        cls.queries = np.ascontiguousarray(rng.standard_normal((2, DIM)).astype(np.float32))
        cls.tmpdir = tempfile.mkdtemp()
        cls.data_file = os.path.join(cls.tmpdir, "data.bin")
        cls.data.tofile(cls.data_file)

    @classmethod
    def tearDownClass(cls):
        os.remove(cls.data_file)
        os.rmdir(cls.tmpdir)

    def build(self, name, metric):
        index = getattr(daisy, name)(metric)
        if hasattr(index, "setWarpingWindow"):
            index.setWarpingWindow(3)
        if name == "ParIS":
            index.buildIndex(self.data_file, DIM, N)
        else:
            index.buildIndex(self.data)
        return index

    def test_k_greater_than_n_raises(self):
        names = [n for n in ("BruteForceSearch", "LbBruteforce", "Messi", "ParIS", "Fresh", "DumpyOS")
                 if hasattr(daisy, n)]
        for name in names:
            for metric in (DistanceType.L2_SQUARED, DistanceType.DTW):
                with self.subTest(algorithm=name, metric=metric):
                    index = self.build(name, metric)
                    with self.assertRaises(ValueError):
                        index.searchIndex(self.queries, N + 1)
                    if hasattr(daisy, "SearchConfig"):
                        config = daisy.SearchConfig()
                        config.k = N + 1
                        with self.assertRaises(ValueError):
                            index.searchIndex(self.queries, config)
                    ids, _ = index.searchIndex(self.queries, N)
                    self.assertEqual(sorted(ids[0].astype(np.int64).tolist()), list(range(N)))

    def test_streaming_insert_raises_the_limit(self):
        index = daisy.Messi(DistanceType.L2_SQUARED)
        index.buildIndex(self.data[: N // 2])
        with self.assertRaises(ValueError):
            index.searchIndex(self.queries, N // 2 + 1)
        index.insertBatch(self.data[N // 2:])
        ids, _ = index.searchIndex(self.queries, N)
        self.assertEqual(len(np.unique(ids[0])), N)


if __name__ == "__main__":
    unittest.main()
