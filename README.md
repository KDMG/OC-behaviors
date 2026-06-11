# OC-Behaviors

*Object-Centric Behaviors mining* on Object-Centric Event Logs (OCEL).

## Requirements

- [Python 3.9](https://www.python.org/downloads/release/python-396/)
- [Graphviz](https://graphviz.org/download/)

## Installation

```bash
git clone https://github.com/KDMG/OC-behaviors.git
cd OC-behaviors

python3.9 -m venv venv
source venv/bin/activate       # Linux / macOS
# venv\Scripts\activate        # Windows

pip install -r requirements.txt --no-dependencies
```

## Quick Start

Run the full pipeline on any [dataset](https://github.com/KDMG/OC-behaviors/tree/main/datasets):

```bash
python pipeline.py datasets/order_management/order-management.sqlite \
    --leading orders \
    --bundle datasets/order_management/mined_behaviors.pkl \
    --explore
```

This will run MLPA to detect resource types, filter the log, mine behaviors with gspan, and open the interactive explorer at `http://127.0.0.1:8050/`.

To open the explorer on an already-mined bundle:

```bash
python pipeline.py --explore-only datasets/order_management/mined_behaviors.pkl
```

## Pipeline Parameters

**Positional:** `ocel` — input OCEL path (`.sqlite`). Not required with `--explore-only`.

**MLPA**
- `--tau` (default: `0.9`) - mlpaDiscovery tau threshold.
- `--remove-types TYPE [TYPE ...]` — object types to remove; auto-detected from MLPA if omitted.
- `--skip-mlpa` — skip the MLPA step (requires `--remove-types` or `--skip-filter`).

**Filter**
- `--filtered PATH` — output path for the filtered OCEL (default: `<input>_filtered.sqlite`).
- `--skip-filter` — skip filtering and mine the raw input directly.

**Mining**
- `--bundle PATH` — output `.pkl` bundle for the explorer.
- `--leading TYPE` — leading object type for execution extraction (required).
- `--miner` (default: `gspan`) — subgraph miner; choices: `gspan`.
- `--kpi KPI` (default: `duration`, `n_events`) — KPI(s) to compute.
- `--s-min` (default: `5%` of process executions) — minimum behavior support.
- `--s-max` (default: `80%` of process executions) — maximum behavior support.
- `--support-abs` — interpret `--s-min`/`--s-max` as absolute counts.
- `--quiet` — silence mining logs.

**Explorer**
- `--explore` — launch the explorer automatically after mining.
- `--host` (default: `127.0.0.1`) — explorer server host.
- `--port` (default: `8050`) — explorer server port.

## Contact

Chiara Gobbi — [c.gobbi@pm.univpm.it](mailto:c.gobbi@pm.univpm.it)
