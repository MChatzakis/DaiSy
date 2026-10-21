"""Exact physical results after bounded disk flushes and LSM compactions."""
import gc
import math
from pathlib import Path
import tempfile
import unittest
import numpy as np
import daisy


@unittest.skipUnless(hasattr(daisy, 'Coconut'), 'Coconut is disabled')
class CoconutDiskTest(unittest.TestCase):
    def check_topk(self, index, data, query):
        exact = np.sum((data.astype(np.float64)[None] - query.astype(np.float64)[:, None]) ** 2, axis=2)
        for k in (1, 10, min(100, len(data)), min(1000, len(data))):
            ids, distances = index.searchIndex(query, k)
            self.assertTrue(np.all(ids < len(data)))
            self.assertTrue(all(len(set(row)) == k for row in ids))
            physical = np.take_along_axis(exact, ids.astype(np.int64), axis=1)
            expected = np.sort(exact, axis=1)[:, :k]
            allowance = 2 * np.abs(np.spacing(expected.astype(np.float32)).astype(np.float64))
            self.assertTrue(np.all(np.abs(np.sort(physical, axis=1) - expected) <= allowance))
            np.testing.assert_allclose(distances, physical, rtol=2e-6, atol=2e-5)

    def test_many_flushes_merges_partial_buffers_and_stable_ids(self):
        rng = np.random.default_rng(677)
        data = rng.normal(size=(777, 96)).astype(np.float32)
        queries = np.concatenate((data[[0, 91, 512, 776]], rng.normal(size=(5, 96)).astype(np.float32)))
        index = daisy.Coconut(daisy.DistanceType.L2_SQUARED)
        index.setBufferCapacity(23)
        index.setLeafSize(7)
        index.setNumThreads(3)
        index.buildIndex(data[:91])
        previous = 91
        for n in (91, 199, 402, 777):
            if n > previous:
                added = data[previous:n].copy()
                index.insertBatch(added)
                added.fill(999)
            index.waitForMerges()
            self.assertEqual(index.getNDatabase(), n)
            self.assertLess(index.getBufferedRecords(), 23)
            self.assertLessEqual(index.getPeakBufferedRecords(), 23)
            self.assertEqual(index.getDiskRecords() + index.getBufferedRecords(), n)
            self.assertEqual(index.getDiskBytes(), index.getDiskRecords() * (96 * 4 + 16 * 2 + 8))
            self.assertLessEqual(index.getRunCount(), 1 + math.ceil(math.log2(index.getFlushCount() + 1)))
            actual_bytes = sum(p.stat().st_size for p in Path(index.getIndexDirectory()).iterdir())
            self.assertEqual(actual_bytes, index.getDiskBytes())
            self.check_topk(index, data[:n], queries)
            previous = n
        self.assertGreater(index.getMergeCount(), 0)
        index.flush()
        index.waitForMerges()
        self.assertEqual(index.getBufferedRecords(), 0)
        self.assertEqual(index.getDiskRecords(), len(data))
        self.check_topk(index, data, queries)

    def test_file_build_odd_key_size_range_and_cleanup(self):
        rng = np.random.default_rng(99)
        # Integer pairs give exact float32 sums and inclusive-radius boundaries.
        data = rng.integers(-3, 4, size=(197, 12)).astype(np.float32)
        queries = data[[0, 49, 110, 196]].copy()
        with tempfile.TemporaryDirectory() as parent:
            source = Path(parent) / 'source.bin'
            data[:49].tofile(source)
            original = source.read_bytes()
            index = daisy.Coconut(daisy.DistanceType.L2_SQUARED)
            index.setIndexDirectory(parent)
            index.setBufferCapacity(13)
            index.setPaaSegments(3)
            index.setLeafSize(5)
            index.buildIndex(str(source), 12)
            index.insert(data[49])
            index.insertBatch(data[50:])
            self.assertEqual(source.read_bytes(), original)
            self.check_topk(index, data, queries)
            for radius in (0., 16., 10000.):
                config = daisy.SearchConfig(); config.type = daisy.QueryType.RANGE; config.r = radius
                ids, distances = index.searchIndex(queries, config)
                for q, row_i, row_d in zip(queries, ids, distances):
                    exact = np.sum((data.astype(np.float64) - q) ** 2, axis=1)
                    self.assertEqual(set(row_i), set(np.flatnonzero(exact <= radius)))
                    np.testing.assert_array_equal(row_d, np.sort(exact[exact <= radius]))
            directory = Path(index.getIndexDirectory())
            del index; gc.collect()
            self.assertFalse(directory.exists())
            self.assertTrue(source.exists())

    def test_strided_arrays_shapes_and_nonfinite_insert(self):
        data = np.arange(600, dtype=np.float32).reshape(50, 12) / 100
        index = daisy.Coconut(daisy.DistanceType.L2_SQUARED)
        index.setPaaSegments(3); index.setBufferCapacity(7)
        index.buildIndex(data[:20:2])
        index.insertBatch(data[21::2])
        expected = np.concatenate((data[:20:2], data[21::2]))
        self.check_topk(index, expected, data[::7])
        n = index.getNDatabase()
        with self.assertRaisesRegex(RuntimeError, 'dimension mismatch'): index.insertBatch(np.zeros((2, 11), np.float32))
        with self.assertRaisesRegex(RuntimeError, 'dimension mismatch'): index.insert(np.zeros(11, np.float32))
        with self.assertRaisesRegex(RuntimeError, 'dimension mismatch'): index.searchIndex(np.zeros((1, 11), np.float32), 1)
        with self.assertRaisesRegex(RuntimeError, 'finite'): index.insert(np.full(12, np.nan, np.float32))
        self.assertEqual(index.getNDatabase(), n)
        with self.assertRaisesRegex(RuntimeError, 'cannot change'): index.setBufferCapacity(3)

    def test_corrupt_directory_rejected_before_mapping_access(self):
        index = daisy.Coconut(daisy.DistanceType.L2_SQUARED)
        data = np.zeros((20, 64), np.float32)
        index.setBufferCapacity(7); index.buildIndex(data)
        next(Path(index.getIndexDirectory()).glob('*.meta')).write_bytes(b'')
        with self.assertRaisesRegex(RuntimeError, 'index leaf/run'): index.searchIndex(data[:1], 1)

    def test_paper_sims_resident_summaries_and_multilevel_tree(self):
        # >128**2 leaves exercises three packed B+tree levels, duplicate keys,
        # both query-key endpoints, and external-sort fan-in beyond 64 runs.
        rng = np.random.default_rng(718)
        data = rng.integers(-4, 5, size=(33001, 12)).astype(np.float32)
        query = np.concatenate((data[[0, 1024, 33000]], np.full((1, 12), -30, np.float32),
                                np.full((1, 12), 30, np.float32)))
        index = daisy.Coconut(daisy.DistanceType.L2_SQUARED)
        index.setPaaSegments(3); index.setLeafSize(2); index.setBufferCapacity(101)
        index.buildIndex(data)
        self.assertEqual(index.getRunCount(), 1)
        self.assertEqual(index.getResidentSummaryBytes(), len(data) * 3)
        exact = np.sum((query.astype(np.float64)[:, None] - data.astype(np.float64)) ** 2, axis=2)
        expected = np.argsort(exact, axis=1, kind='stable')[:, :10]
        for threads in (1, 4):
            index.setNumThreads(threads)
            ids, distances = index.searchIndex(query, 10)
            np.testing.assert_array_equal(ids, expected)
            np.testing.assert_array_equal(distances, np.take_along_axis(exact, expected, axis=1))
            stats = index.getLastSearchStats()
            self.assertEqual(stats['summaries_examined'], len(data) * len(query))
            self.assertEqual(stats['runs_searched'], len(query))
            self.assertGreater(stats['raw_bytes_read'], 0)
            self.assertGreater(stats['bounds_seconds'], 0)

    def test_paper_async_compaction_live_snapshot_and_barrier(self):
        rng = np.random.default_rng(270)
        data = rng.integers(-3, 4, size=(2049, 12)).astype(np.float32)
        query = data[[1, 127, 777, 2048]].copy()
        for async_merges in (False, True):
            with self.subTest(async_merges=async_merges):
                index = daisy.Coconut(daisy.DistanceType.L2_SQUARED)
                index.setPaaSegments(3); index.setBufferCapacity(17); index.setLeafSize(9)
                index.setAsyncMerges(async_merges); index.setNumThreads(3)
                index.buildIndex(data[:131])
                previous = 131
                for n in (198, 513, 1028, 2049):
                    index.insertBatch(data[previous:n])
                    # No barrier: queries must be exact even if old runs are
                    # being merged/replaced concurrently by the worker.
                    exact = np.sum((query[:, None].astype(np.float64) - data[None, :n]) ** 2, axis=2)
                    expected = np.argsort(exact, axis=1, kind='stable')[:, :10]
                    ids, distances = index.searchIndex(query, 10)
                    np.testing.assert_array_equal(ids, expected)
                    np.testing.assert_array_equal(distances, np.take_along_axis(exact, expected, axis=1))
                    previous = n
                index.flush(); index.waitForMerges()
                self.assertEqual(index.getResidentSummaryBytes(), len(data) * 3)
                self.assertEqual(index.getDiskBytes(), sum(p.stat().st_size for p in Path(index.getIndexDirectory()).iterdir()))
                self.assertLessEqual(index.getRunCount(), 1 + math.ceil(math.log2(index.getFlushCount() + 1)))
                with self.assertRaisesRegex(RuntimeError, 'cannot change'): index.setAsyncMerges(False)
                directory = Path(index.getIndexDirectory())
                del index; gc.collect()
                self.assertFalse(directory.exists())

    def test_paper_btp_recent_windows_skip_old_runs_and_preserve_ids(self):
        rng = np.random.default_rng(102)
        data = rng.integers(-2, 3, size=(773, 12)).astype(np.float32)
        data[-1] = data[0]  # newest exact match must win inside a small window
        query = data[[0, 20, 772]].copy()
        index = daisy.Coconut(daisy.DistanceType.L2_SQUARED)
        index.setPaaSegments(3); index.setLeafSize(7); index.setBufferCapacity(16)
        index.setNumThreads(4); index.buildIndex(data[:128]); index.insertBatch(data[128:]); index.waitForMerges()
        for window in (1, 5, 17, 200, 773, 1000):
            first = max(0, len(data) - window)
            exact = np.sum((query[:, None].astype(np.float64) - data[None, first:]) ** 2, axis=2)
            expected = np.argsort(exact, axis=1, kind='stable')[:, :10] + first
            ids, distances = index.searchIndexRecent(query, 10, window)
            count = min(10, len(data) - first)
            np.testing.assert_array_equal(ids[:, :count], expected)
            np.testing.assert_array_equal(distances[:, :count], np.take_along_axis(exact, expected - first, axis=1))
            self.assertTrue(np.all(distances[:, count:] == np.finfo(np.float32).max))
            if window <= 17:
                self.assertGreater(index.getLastSearchStats()['runs_skipped'], 0)
        with self.assertRaisesRegex(RuntimeError, 'window must be positive'):
            index.searchIndexRecent(query, 1, 0)
        with self.assertRaisesRegex(RuntimeError, 'dimension mismatch'):
            index.searchIndexRecent(np.zeros((1, 11), np.float32), 1, 5)

    def test_batch_sims_matches_serial_and_exact_with_tail_and_windows(self):
        rng = np.random.default_rng(1120)
        data = rng.integers(-3, 4, size=(2049, 12)).astype(np.float32)
        query = data[rng.integers(0, len(data), size=65)].copy()
        index = daisy.Coconut(daisy.DistanceType.L2_SQUARED)
        index.setPaaSegments(3); index.setBufferCapacity(97); index.setLeafSize(17)
        index.setNumThreads(3); index.buildIndex(data[:513]); index.insertBatch(data[513:])
        for window in (len(data), 29):
            first = max(0, len(data) - window)
            exact = np.sum((query[:, None].astype(np.float64) - data[None, first:]) ** 2, axis=2)
            for batch_size in (1, 4, 32, 64):
                index.setQueryBatchSize(batch_size)
                for k in (1, 10, 31):
                    ids, distances = index.searchIndexRecent(query, k, window)
                    n = min(k, len(data) - first)
                    expected = np.argsort(exact, axis=1, kind='stable')[:, :n]
                    np.testing.assert_array_equal(ids[:, :n], expected + first)
                    np.testing.assert_array_equal(distances[:, :n], np.take_along_axis(exact, expected, axis=1))
                    self.assertTrue(np.all(distances[:, n:] == np.finfo(np.float32).max))
                    stats = index.getLastSearchStats()
                    self.assertLessEqual(stats['distance_evaluations'] - stats['seed_evaluations'], stats['candidate_records'])
                    self.assertLessEqual(stats['unique_candidate_records'], stats['raw_records_read'])
                    self.assertLess(stats['peak_scratch_bytes'], 20 * 1024 * 1024)
        for bad in (0, 65):
            with self.assertRaisesRegex(RuntimeError, 'between 1 and 64'): index.setQueryBatchSize(bad)

    def test_batch_sims_shares_reads_without_replacing_sax_pruning(self):
        rng = np.random.default_rng(728)
        data = rng.normal(size=(4096, 12)).astype(np.float32)
        query = np.repeat(np.zeros((1, 12), np.float32), 17, axis=0)
        index = daisy.Coconut(daisy.DistanceType.L2_SQUARED)
        index.setPaaSegments(3); index.setLeafSize(17); index.setBufferCapacity(512)
        index.setNumThreads(4); index.buildIndex(data)
        index.setQueryBatchSize(1); before = index.searchIndex(query, 10); serial = index.getLastSearchStats()
        index.setQueryBatchSize(32); after = index.searchIndex(query, 10); batched = index.getLastSearchStats()
        np.testing.assert_array_equal(before[0], after[0]); np.testing.assert_array_equal(before[1], after[1])
        self.assertLess(batched['raw_bytes_read'], serial['raw_bytes_read'] / 4)
        self.assertEqual(batched['candidate_records'], serial['candidate_records'])
        # An exact seed prunes the distant cluster: no unconditional raw scan.
        data.fill(1); data[0].fill(0); index.buildIndex(data)
        ids, distances = index.searchIndex(query, 1)
        np.testing.assert_array_equal(ids, np.zeros((17, 1), np.uint64))
        np.testing.assert_array_equal(distances, np.zeros((17, 1), np.float32))
        stats = index.getLastSearchStats()
        self.assertEqual(stats['candidate_records'], 0)
        self.assertEqual(stats['distance_evaluations'], stats['seed_evaluations'])
        self.assertLess(stats['raw_bytes_read'], len(data) * 12 * 4)


if __name__ == '__main__':
    unittest.main()
