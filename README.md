# OC-Behaviors

A tool for **object-centric behaviors mining* on Object-Centric Event Logs (OCEL).

Given an OCEL, the tool automatically discovers a hierarchy of object types, filters out resource and container types, and then mines frequent subgraph patterns at each level of abstraction — scoring them by their discriminative power with respect to a user-defined KPI (e.g., case duration, number of events).

## Requirements

- [Python 3.9.6](https://www.python.org/downloads/release/python-396/)
- [Graphviz](https://graphviz.org/download/) (system package, required for graph rendering)

All Python dependencies are listed in `requirements.txt`:

```
networkx, numpy, pandas, pm4py, ocpa, gspan-mining,
scikit-learn, scipy, Flask, Jinja2, graphviz, pulp
```

---

## Installation

```bash
git clone https://github.com/<your-org>/OC-behaviors.git
cd OC-behaviors

# Create and activate a virtual environment (recommended)
python3.9 -m venv .venv
source .venv/bin/activate       # Linux / macOS
# .venv\Scripts\activate        # Windows

pip install -r requirements.txt
```

> **Note on Graphviz:** the Python `graphviz` package is a wrapper — you also need the Graphviz binaries installed at the system level.
> - macOS: `brew install graphviz`
> - Ubuntu/Debian: `sudo apt install graphviz`
> - Windows: download from [graphviz.org](https://graphviz.org/download/) and add to PATH.

---

## Quick Start

Run the full pipeline on the provided **Hinge Production** dataset with a single command:

```bash
python pipeline.py datasets/hinge_production/hinge_production.sqlite \
    --leading Hinge \
    --bundle  datasets/hinge_production/run_gspan.pkl \
    --explore
```

This will:
1. Run MLPA to detect resource types automatically.
2. Filter the log (removing types with no process activities).
3. Mine behavioral patterns with gspan (default miner).
4. Open the interactive explorer at `http://127.0.0.1:8050/`.

---
