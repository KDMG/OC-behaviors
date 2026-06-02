# OC-Behaviors

A tool for **multi-level behavioral pattern mining** on Object-Centric Event Logs (OCEL).

Given an OCEL, the tool automatically discovers a hierarchy of object types, filters out resource and container types, and then mines frequent subgraph patterns at each level of abstraction — scoring them by their discriminative power with respect to a user-defined KPI (e.g., case duration, number of events).

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Requirements](#requirements)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Pipeline in Detail](#pipeline-in-detail)
  - [Step 1 — Multi-Level Process Analysis (MLPA)](#step-1--multi-level-process-analysis-mlpa)
  - [Step 2 — Log Filtering](#step-2--log-filtering)
  - [Step 3 — ρ-Configured Pattern Mining](#step-3--ρ-configured-pattern-mining)
  - [Step 4 — Interactive Explorer](#step-4--interactive-explorer)
- [Running the Full Pipeline](#running-the-full-pipeline)
- [Running Individual Steps](#running-individual-steps)
- [Datasets](#datasets)
- [Project Structure](#project-structure)

---

## Overview

Object-Centric Process Mining operates on logs where events relate to multiple objects of different types simultaneously. A key challenge is understanding *which* object types carry process-level semantics (i.e., participate in activities) and which are purely structural resources or containers.

This tool addresses that challenge through a four-step pipeline:

1. **MLPA** assigns each object type to a level in a resource hierarchy.
2. **Filtering** removes resource types (those with no process activities) and absorbs their attributes into the remaining objects.
3. **ρ-lift mining** enumerates configurations (ρ) of abstraction levels across object types, extracts process executions, and for each configuration mines frequent subgraph patterns ranked by KPI lift — i.e., how strongly a pattern discriminates high-KPI from low-KPI executions.
4. **Explorer** is a browser-based interactive interface to inspect patterns, filter by configuration, and drill down into individual process executions.

---

## Architecture

```
OCEL (.sqlite)
      │
      ▼
┌─────────────┐   levels_dict + process_view_with_events
│    MLPA     │  ──────────────────────────────────────────► auto-detected
│ (MLPAMiner) │                                              resource types
└─────────────┘
      │
      ▼ resource types to remove
┌─────────────┐
│  ocel_filter│  filtered OCEL (.sqlite)
│             │ ──────────────────────────────────────────►
└─────────────┘
      │
      ▼ filtered OCEL
┌─────────────┐
│  rho_lift   │  per object-type, sweep ρ-abstraction levels;
│             │  extract process executions → mine subgraph patterns;
│             │  score by KPI lift (Δ) → .pkl bundle
└─────────────┘
      │
      ▼ bundle.pkl
┌─────────────┐
│ oc_explorer │  Flask app — browse patterns, configs, executions
│  (browser)  │
└─────────────┘
```

---

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

## Pipeline in Detail

### Step 1 — Multi-Level Process Analysis (MLPA)

MLPA ([Gobbi et al., 2024](https://www.google.com/search?q=model+repair+supported+by)) analyses the temporal and cardinality relationships between object types across the event log and assigns each type to a level in a resource hierarchy.

The key output is a partition of object types into levels. Types that appear at a level with an **empty activity set** — i.e., they participate in events but are not the process-level subjects — are identified as *resource types* and can be safely removed before pattern mining.

**Output example** (Hinge Production log):

```
{
  0.0: [(['SteelSheet'], {'SplitSteelSheet', 'FormSteelSheet', 'HeatSteelSheet'}),
        (['SteelPin'], {'AssembleHinge'}),
        (['HingePack'], {'PackHinges'})],
  1.0: [(['MalePart', 'FormedPart', 'FemalePart', 'Hinge'], {<activities>}),
        (['SteelCoil'], set())],      ← no activities → resource type
  2.0: [(['Machine'], set())],        ← no activities → resource type
  3.0: [(['Facility'], {'MoveParts'}),
        (['Workstation'], set())],    ← no activities → resource type
  4.0: [(['Worker'], set())]          ← no activities → resource type
}
```

In this example, `SteelCoil`, `Machine`, `Workstation`, and `Worker` are automatically detected as resource types.

---

### Step 2 — Log Filtering

The filtering step (`ocel_filter.py`) removes resource object types from the log and **absorbs their attributes** into the connected process-level objects, so no information is lost. It also:

- Sanitises object type names (spaces → underscores, lowercase).
- Sanitises activity names (removes special characters).
- Discretises numeric attributes automatically using k-means clustering (elbow method).

```bash
python ocel_filter.py auto <input.sqlite> <output.sqlite> <TypeToRemove1> [TypeToRemove2 ...]
```

---

### Step 3 — ρ-Configured Pattern Mining

`rho_lift.py` is the core mining engine. For each object type it defines a set of *abstraction levels* (ρ), e.g., `identity` (each object is distinct), `type` (all objects of the same type are merged), or intermediate levels derived from object attributes. A **configuration** Φ assigns one level to each object type.

For each configuration Φ:
1. Process executions are extracted (leading-type extraction via ocpa).
2. Each execution is encoded as a labelled directed graph (behavior graph).
3. Frequent subgraphs are mined with **gspan** (or subdue).
4. Each pattern is scored by **KPI lift** Δ: the difference in mean KPI between executions containing the pattern and those that do not.

Results are saved to a `.pkl` bundle containing all configurations, all patterns, and all KPI statistics.

```bash
python rho_lift.py <filtered.sqlite> \
    --leading   <leading_type> \
    --miner     gspan \
    --kpi       duration \
    --kpi       n_events \
    --s-min     <min_support> \
    --s-max     <max_support> \
    --support-abs \
    --bundle    <output.pkl>
```

**Key parameters**

| Parameter | Description | Default |
|---|---|---|
| `--leading` | Object type used as case notion for execution extraction | *(required)* |
| `--miner` | Subgraph miner: `gspan` or `subdue` | `gspan` |
| `--kpi` | KPI(s) to compute. Repeatable. Built-ins: `duration`, `n_events`, `n_objects`, `n_activities`, `event_density`. Custom: `name:=EXPR`. | `duration` |
| `--s-min` | Minimum pattern support | 5% of executions |
| `--s-max` | Maximum pattern support | 80% of executions |
| `--support-abs` | Treat `--s-min`/`--s-max` as absolute counts | auto-enabled with defaults |
| `--bundle` | Output `.pkl` path | *(required)* |
| `--exclude-attrs` | Comma-separated attribute names to exclude from ρ-space | — |
| `--top-k` | Top-k patterns to keep globally | 10 |
| `--quiet` | Suppress progress output | false |

---

### Step 4 — Interactive Explorer

The explorer (`explorer/oc_explorer.py`) is a browser-based Flask application for interactive analysis of the mining results stored in a `.pkl` bundle.

Features:
- Browse the **ρ-configuration lattice** and filter by interestingness.
- Inspect individual **patterns** with their KPI lift, support, and graph visualisation.
- Drill down into specific **process executions** and compare their behavior graphs.
- Attribute breakdown and cross-configuration analysis.

```bash
python explorer/oc_explorer.py <bundle.pkl> [--port 8050]
```

Then open `http://127.0.0.1:8050/` in your browser.

---

## Running the Full Pipeline

`pipeline.py` chains all four steps into a single command.

### Full pipeline (auto-detect resource types)

```bash
python pipeline.py <input.sqlite> \
    --leading  <leading_type> \
    --bundle   <output.pkl> \
    [--miner   gspan|subdue] \
    [--kpi     duration] [--kpi n_events] \
    [--s-min   N] [--s-max N] [--support-abs] \
    [--tau     0.9] \
    [--explore]
```

### Common options

| Option | Description |
|---|---|
| `--leading TYPE` | Leading object type (required) |
| `--bundle PATH` | Output `.pkl` bundle (required) |
| `--tau FLOAT` | MLPA threshold (default: `0.9`) |
| `--remove-types T1 T2 …` | Override auto-detected types to remove |
| `--skip-mlpa` | Skip MLPA step entirely |
| `--filtered PATH` | Custom path for filtered OCEL (default: `<input>_filtered.sqlite`) |
| `--skip-filter` | Mine the raw OCEL without filtering |
| `--miner gspan\|subdue` | Subgraph miner (default: `gspan`) |
| `--kpi KPI` | KPI to compute; repeatable (default: `duration n_events`) |
| `--s-min N` | Min support (default: 5% of process executions, absolute) |
| `--s-max N` | Max support (default: 80% of process executions, absolute) |
| `--support-abs` | Treat support bounds as absolute counts |
| `--explore` | Launch explorer automatically after mining |
| `--host HOST` | Explorer host (default: `127.0.0.1`) |
| `--port PORT` | Explorer port (default: `8050`) |
| `--quiet` | Silence progress output |

### Examples

```bash
# Hinge Production — full pipeline, then explore
python pipeline.py datasets/hinge_production/hinge_production.sqlite \
    --leading Hinge --bundle datasets/hinge_production/run_gspan.pkl \
    --explore

# Order-to-Cash — manual resource types, absolute support
python pipeline.py datasets/order_to_cash/o2c.sqlite \
    --leading Order \
    --remove-types Customer Product \
    --bundle datasets/order_to_cash/run_gspan.pkl \
    --kpi duration --kpi n_events \
    --s-min 50 --s-max 1200 --support-abs

# Purchase-to-Pay — skip MLPA, use pre-filtered log
python pipeline.py datasets/purchase_to_pay/ocel2-p2p_filtered.sqlite \
    --skip-mlpa --skip-filter \
    --leading purchase_order \
    --bundle datasets/purchase_to_pay/run_gspan.pkl

# Open an existing bundle without re-running the pipeline
python pipeline.py --explore-only datasets/hinge_production/run_gspan.pkl
```

---

## Running Individual Steps

You can run each step independently if needed.

### MLPA only

```python
from mlpa.my_ocel_importer import apply as load_ocel
from mlpa.MLPAMiner import mlpaDiscovery

ocel = load_ocel("datasets/hinge_production/hinge_production.sqlite")
levels, process_view, process_view_with_events = mlpaDiscovery(ocel, tau=0.9)
```

### Filter only

```bash
python ocel_filter.py auto \
    datasets/hinge_production/hinge_production.sqlite \
    datasets/hinge_production/hinge_production_filtered.sqlite \
    Worker Workstation Machine SteelCoil
```

### Mine only

```bash
python rho_lift.py datasets/hinge_production/hinge_production_filtered.sqlite \
    --leading Hinge --miner gspan \
    --kpi duration --kpi n_events \
    --s-min 145 --s-max 2326 --support-abs \
    --bundle datasets/hinge_production/run_gspan.pkl --quiet
```

### Explorer only

```bash
python explorer/oc_explorer.py datasets/hinge_production/run_gspan.pkl --port 8050
```

---

## Datasets

The `datasets/` folder contains six Object-Centric Event Logs in OCEL 2.0 SQLite format, each pre-processed with a pre-built `.pkl` bundle so the explorer can be launched immediately.

| Dataset | Domain | Objects | Events | Object Types | Activities |
|---|---|---|---|---|---|
| `hinge_production` | Manufacturing (hinge assembly) | 23 771 | 38 528 | 12 | 11 |
| `order_to_cash` | Order-to-Cash (O2C) | 8 819 | 28 278 | 9 | 22 |
| `order_management` | E-commerce order management | 10 840 | 21 008 | 6 | 11 |
| `purchase_to_pay` | Purchase-to-Pay (P2P) | 9 543 | 14 671 | 7 | 10 |
| `container_logistics` | Container logistics & shipping | 13 882 | 35 372 | 7 | 14 |
| `transfer_order` | Warehouse transfer orders (SAP) | 2 500 | 10 319 | 5 | 3 |

To explore any dataset immediately without re-running the pipeline:

```bash
python explorer/oc_explorer.py datasets/hinge_production/run_gspan.pkl
python explorer/oc_explorer.py datasets/order_to_cash/run_gspan.pkl
python explorer/oc_explorer.py datasets/order_management/run_gspan.pkl
python explorer/oc_explorer.py datasets/purchase_to_pay/run_gspan.pkl
python explorer/oc_explorer.py datasets/container_logistics/run_gspan.pkl
python explorer/oc_explorer.py datasets/transfer_order/run_gspan.pkl
```

---

## Project Structure

```
OC-behaviors/
│
├── pipeline.py            # End-to-end pipeline (MLPA → filter → mine → explore)
├── rho_lift.py            # ρ-configured pattern mining CLI
├── ocel_filter.py         # OCEL filtering and attribute absorption
├── requirements.txt
│
├── mining/                # Core mining engine
│   ├── core.py            # BehaviorGraph, ProcessExecution, OCELLevelGraph
│   ├── kpi.py             # KPI definitions and computation
│   ├── mining_gspan.py    # gspan subgraph mining adapter
│   └── subdue_experiment.py
│
├── explorer/              # Interactive browser-based explorer
│   ├── oc_explorer.py     # Flask app entry point
│   ├── oc_state.py        # Explorer state management
│   ├── oc_attrs.py        # Attribute breakdown utilities
│   └── oc_tree_analysis.py
│
├── mlpa/                  # Multi-Level Process Analysis
│   ├── MLPAMiner.py       # mlpaDiscovery: level assignment via MILP
│   ├── totmminer.py       # Temporal-cardinality relation mining
│   ├── select_level.py    # Process execution utilities
│   └── my_ocel_importer.py
│
├── ocpa/                  # Vendored ocpa library (OCEL processing)
│
└── datasets/
    ├── hinge_production/
    ├── order_to_cash/
    ├── order_management/
    ├── purchase_to_pay/
    ├── container_logistics/
    └── transfer_order/
```
