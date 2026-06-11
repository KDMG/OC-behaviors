# OC-Behaviors

*Object-Centric Behaviors mining* on Object-Centric Event Logs (OCEL).

## Requirements

- [Python 3.9.23](https://www.python.org/downloads/release/python-396/) (any other python 3.9 and 3.10 should work as well)
- [Graphviz](https://graphviz.org/download/)

## Installation

```bash
git clone https://github.com/KDMG/OC-behaviors.git
cd OC-behaviors

python3.9 -m venv .venv
source .venv/bin/activate       # Linux / macOS
# .venv\Scripts\activate        # Windows

pip install -r requirements.txt --no-dependencies
```

## Quick Start

Run the full pipeline on any [dataset](https://github.com/KDMG/OC-behaviors/tree/main/datasets). For example:

```bash
python pipeline.py datasets/order_management/order-management.sqlite \
    --leading order \
    --bundle datasets/order_management/mined_behaviors.pkl \
    --explore
```

This will:
1. Run MLPA to detect resource types automatically;
2. Filter the log (removing types with no process activities);
3. Mine behavioral patterns with gspan;
4. Open the interactive explorer at `http://127.0.0.1:8050/`;

To explore already mined behaviors run:
```bash
python pipeline.py --explore-only datasets/order_management/mined_behaviors.pkl
```


