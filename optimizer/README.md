# Knob optimizer: runs and results

| Folder | Content |
|---|---|
| [`slurm/`](slurm/README.md) | Slurm jobs for the stella cluster: build the collector, collect one dataset per node, prepare the datasets |
| `results/<server>/` | The collector's output on a server: `samples_<DATASET>.csv` (one per dataset, same header) and `logs/` (the Slurm output of every job). Not versioned (see `results/.gitignore`) |

The collector itself is [`lib/optimizer/optimizer_collector.cpp`](../lib/optimizer/optimizer_collector.cpp),
documented in [docs/optimizer-notes/collector.md](../docs/optimizer-notes/collector.md); the
datasets are prepared by [scripts/datasets/prepare_datasets.py](../scripts/datasets/prepare_datasets.py)
([docs/optimizer-notes/datasets.md](../docs/optimizer-notes/datasets.md)).
