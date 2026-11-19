#!/usr/bin/env python3
"""Builds the SALD data-series dataset from the raw SALD fMRI archives.

SALD (Southwest University Adult Lifespan Dataset, 494 subjects; license CC BY-NC,
cite Wei et al. 2017, bioRxiv 177279) has one resting-state fMRI scan per subject:
64 x 64 x 32 voxels x 242 time points (TR 2 s). Download it with download_sald.sh.

Every brain voxel's time course is cut into windows of 128 points with stride 8
(15 windows per voxel), and every window is z-normalized. This gives about 200M series,
the size of the SALD data-series benchmark (200M series of length 128, about 100 GB)
used in the MESSI / SING / Odyssey papers; their exact procedure is not published, so
this file is comparable in size and source, not guaranteed identical.

The first subject read (the first in the first archive by name) is held out: it only
provides queries, so no query has an overlapping window of itself in the base.

Outputs, raw float32 with 128 values per record (the format of the Seismic and Astro
files), in OUT_DIR:
  data_size200M_sald_len128_znorm.bin     base, every window of every other subject
  queries_size10K_sald_len128_znorm.bin   10,000 random windows of the held-out subject
  data_size200M_sald_len128_znorm.txt     parameters and counts of this build

Usage: python3 -I scripts/datasets/make_sald.py ARCHIVE_DIR OUT_DIR
Needs numpy. The archives are read directly; nothing is extracted to disk.
"""
import gzip
import os
import struct
import sys
import tarfile
import time

import numpy as np

LENGTH = 128      # points per series
STRIDE = 8        # window step
MASK_LEVEL = 0.1  # a voxel is brain if its mean signal exceeds 10% of the scan's highest voxel mean
N_QUERIES = 10000
SEED = 0
BASE_NAME = "data_size200M_sald_len128_znorm.bin"
QUERY_NAME = "queries_size10K_sald_len128_znorm.bin"

NIFTI_TYPES = {2: np.uint8, 4: np.int16, 8: np.int32, 16: np.float32, 64: np.float64, 512: np.uint16}


def voxel_series(nifti_gz):
    """Time series of the brain voxels of one gzipped NIfTI-1 scan: array (voxels, time)."""
    raw = gzip.decompress(nifti_gz)
    if struct.unpack("<i", raw[0:4])[0] != 348:
        raise ValueError("not a little-endian NIfTI-1 file")
    nx, ny, nz, nt = struct.unpack("<8h", raw[40:56])[1:5]
    datatype = struct.unpack("<h", raw[70:72])[0]
    offset = int(struct.unpack("<f", raw[108:112])[0])
    slope, inter = struct.unpack("<ff", raw[112:120])
    data = np.frombuffer(raw, dtype=NIFTI_TYPES[datatype], count=nx * ny * nz * nt, offset=offset)
    series = data.reshape(nt, nx * ny * nz).T.astype(np.float32)  # file order: x fastest, time slowest
    if slope not in (0.0, 1.0) or inter != 0.0:
        series = series * (slope if slope != 0.0 else 1.0) + inter
    mean, std = series.mean(axis=1), series.std(axis=1)
    return series[(mean > MASK_LEVEL * mean.max()) & (std > 0)]


def windows(series):
    """z-normalized windows of length LENGTH, step STRIDE: array (n, LENGTH), float32."""
    if series.shape[1] < LENGTH:
        return np.empty((0, LENGTH), np.float32)
    w = np.lib.stride_tricks.sliding_window_view(series, LENGTH, axis=1)[:, ::STRIDE, :].reshape(-1, LENGTH)
    mean = w.mean(axis=1, keepdims=True)
    std = w.std(axis=1, keepdims=True)
    keep = std[:, 0] > 1e-6  # a constant window cannot be z-normalized
    return ((w[keep] - mean[keep]) / std[keep]).astype(np.float32)


def rest_scans(archive_dir):
    """(subject, gzipped NIfTI bytes) of every resting-state scan, archive by archive."""
    for name in sorted(os.listdir(archive_dir)):
        if not name.endswith(".tar.gz"):
            continue
        with tarfile.open(os.path.join(archive_dir, name), "r|gz") as tar:  # stream, no seeking
            for member in tar:
                base = os.path.basename(member.name)
                if member.isfile() and base.endswith("_task-rest_bold.nii.gz"):
                    yield base.split("_")[0], tar.extractfile(member).read()


def main():
    if len(sys.argv) != 3:
        sys.exit("usage: make_sald.py ARCHIVE_DIR OUT_DIR")
    archive_dir, out_dir = sys.argv[1], sys.argv[2]
    os.makedirs(out_dir, exist_ok=True)
    base_path, query_path = os.path.join(out_dir, BASE_NAME), os.path.join(out_dir, QUERY_NAME)
    t0 = time.time()
    held_out, subjects, base_count, query_count = None, 0, 0, 0
    with open(base_path + ".tmp", "wb") as base:
        for subject, scan in rest_scans(archive_dir):
            w = windows(voxel_series(scan))
            if held_out is None:  # the first subject only provides queries
                held_out = subject
                pick = np.random.default_rng(SEED).permutation(len(w))[:N_QUERIES]
                w[pick].tofile(query_path + ".tmp")
                query_count = len(pick)
                print(f"{subject}: held out, {query_count} queries", flush=True)
                continue
            w.tofile(base)
            subjects += 1
            base_count += len(w)
            if subjects % 25 == 0:
                print(f"{subjects} subjects, {base_count:,} series, {time.time() - t0:.0f} s", flush=True)
    os.replace(base_path + ".tmp", base_path)
    os.replace(query_path + ".tmp", query_path)
    info = (f"SALD data series built by scripts/datasets/make_sald.py\n"
            f"length {LENGTH}, stride {STRIDE}, brain mask: voxel mean > {MASK_LEVEL} x highest voxel mean, z-normalized windows\n"
            f"base: {base_count} series from {subjects} subjects ({BASE_NAME})\n"
            f"queries: {query_count} random windows (seed {SEED}) of held-out subject {held_out} ({QUERY_NAME})\n")
    with open(os.path.join(out_dir, BASE_NAME.replace(".bin", ".txt")), "w") as f:
        f.write(info)
    print(info + f"took {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
