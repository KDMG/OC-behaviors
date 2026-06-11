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

This will:
1. Run MLPA to detect resource types automatically;
2. Filter the log (removing types with no process activities);
3. Mine behaviors with gspan;
4. Open the interactive explorer at `http://127.0.0.1:8050/`.

To open the explorer on an already-mined bundle:

```bash
python pipeline.py --explore-only datasets/order_management/mined_behaviors.pkl
```

## Pipeline Parameters

### Positional

| Argument | Description |
|----------|-------------|
| `ocel` | Input OCEL path (`.sqlite`). Not required with `--explore-only`. |

### General

| Argument | Default | Description |
|----------|---------|-------------|
| `--explore-only BUNDLE` | — | Skip everything and open the explorer on an existing `.pkl` bundle. |

### MLPA

| Argument | Default | Description |
|----------|---------|-------------|
| `--tau` | `0.9` | mlpaDiscovery tau threshold. |
| `--remove-types TYPE [TYPE ...]` | auto-detected | Object types to remove. If omitted, inferred from MLPA output. |
| `--skip-mlpa` | `False` | Skip the MLPA step entirely (requires `--remove-types` or `--skip-filter`). |

### Filter

| Argument | Default | Description |
|----------|---------|-------------|
| `--filtered PATH` | `<input>_filtered.sqlite` | Output path for the filtered OCEL. |
| `--skip-filter` | `False` | Skip the filter step and mine the raw input OCEL directly. |

### Mining

| Argument | Default | Description |
|----------|---------|-------------|
| `--bundle PATH` | — | Output `.pkl` bundle for the interactive explorer. |
| `--leading TYPE` | — | Leading object type for execution extraction (required for mining). |
| `--miner` | `gspan` | Subgraph miner to use. Choices: `gspan`, `subdue`. |
| `--kpi KPI` | `duration`, `n_events` | KPI(s) to compute. Repeatable (e.g. `--kpi duration --kpi n_events`). |
| `--s-min` | `5%` of executions | Minimum behavior support. |
| `--s-max` | `80%` of executions | Maximum behavior support. |
| `--support-abs` | `False` | Interpret `--s-min`/`--s-max` as absolute counts instead of fractions. |
| `--quiet` | `False` | Silence mining progress logs. |

### Explorer

| Argument | Default | Description |
|----------|---------|-------------|
| `--explore` | `False` | Launch the interactive explorer automatically after mining. |
| `--host` | `127.0.0.1` | Host for the explorer server. |
| `--port` | `8050` | Port for the explorer server. |
