#!/usr/bin/env bash
# Downloads the raw SALD fMRI archives (Southwest University Adult Lifespan Dataset,
# 494 subjects, about 23 GB) from the public INDI S3 bucket into DIR.
# Source: https://fcon_1000.projects.nitrc.org/indi/retro/sald.html (license CC BY-NC;
# cite Wei et al. 2017, bioRxiv 177279). Convert them with make_sald.py.
#
# Usage: scripts/datasets/download_sald.sh DIR
set -euo pipefail
dir="${1:?usage: download_sald.sh DIR}"
mkdir -p "$dir"
base=https://s3.amazonaws.com/fcp-indi/data/Projects/INDI/SALD/RawData_tar
# The archive names are listed on the SALD page; note the '-' in sub-031474-sub-031523.
for name in sub-031274_sub-031323 sub-031324_sub-031373 sub-031374_sub-031423 sub-031424_sub-031473 \
            sub-031474-sub-031523 sub-031524_sub-031573 sub-031574_sub-031623 sub-031624_sub-031673 \
            sub-031674_sub-031723 sub-031724_sub-031767; do
    echo "downloading $name.tar.gz"
    curl -fsS --retry 5 -C - -o "$dir/$name.tar.gz" "$base/$name.tar.gz"
done
echo "done: $(du -sh "$dir" | cut -f1) in $dir"
