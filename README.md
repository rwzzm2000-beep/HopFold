# HopFold

This is a **small public reference repository** for the HopFold paper. It is intentionally not the full internal research codebase: the repository exposes the core paired-carrier idea and a tiny smoke example, while training orchestration, GPU kernels, preprocessing pipelines, machine-specific configuration, and complete benchmark logs are omitted.

> The code is provided to make the method identifiable and inspectable. It is not intended to reproduce all paper numbers directly.

## What is included

- `hopfold/core.py`: a compact reference implementation of the graph-aligned carrier:
  one byte-sized code per feature channel and a paired codebook entry for (i) the self feature and (ii) a first-hop neighbor statistic.
- `examples/toy_demo.py`: a CPU-only smoke example on a four-node graph.
- `results/toy_smoke.json`: the corresponding minimal metadata record.

The public implementation deliberately excludes the full training runner, large-graph data preparation, CUDA/Triton paths, optimization scripts, and internal evaluation artifacts.

## Quick start

```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
$env:PYTHONPATH = "."       # PowerShell; use `export PYTHONPATH=.` on Linux/macOS
python examples/toy_demo.py
```

## Datasets

The paper experiments use public graph-learning datasets. They are not bundled in this repository. For OGB datasets, install the official `ogb` package and use its documented download path; for DGL datasets, use the corresponding DGL dataset class. Keep downloaded data outside the repository (for example under `data/`) and do not commit raw datasets or generated caches.

Typical OGB entry point:

```python
from ogb.nodeproppred import DglNodePropPredDataset
dataset = DglNodePropPredDataset(name="ogbn-products", root="data/ogb")
```

Please consult the official dataset pages for licenses, citation requirements, splits, and current download instructions before using a dataset in a paper reproduction.

## Reproducibility note

The toy example is a sanity check for the public API, not a benchmark. Exact paper results require the authors' internal preprocessing and experiment configuration. A fuller release can be prepared after the review period if appropriate.

## Citation

The citation entry will be added after the paper metadata and venue version are finalized.

## License

No license is granted yet. Please contact the authors before reusing this code beyond inspection or academic discussion.
