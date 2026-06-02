#!/usr/bin/env python3
"""
subdue_experiment.py
────────────────────
Compare frequent labeled subgraph patterns between the **bottom** (all
dimensions at level 0) and the **top** (all dimensions at their maximum
level) configuration of the OCEL behavior lattice.

For each of those two sets of behavior graphs we run a SUBDUE-style
(beam-search) labeled subgraph miner that respects:

  • node labels  = activity  + multiset of (object-label, cardinality ∈ {1,*})
  • edge labels  = object-label + (κ_s, κ_t) cardinalities

and we emit a single self-contained interactive HTML report with the
top-K patterns side-by-side (Cytoscape.js drawings + support / size).

Usage
-----
    python subdue_experiment.py  <ocel_path>  [options]

Example
-------
    python subdue_experiment.py datasets_full/order-management.sqlite \\
        --k 10 --max-edges 4 --beam 12 --min-support 2 \\
        --out subdue_report.html
"""
from __future__ import annotations

import argparse
import html
import itertools
import json
import os
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, Iterable, List, Optional, Sequence, Tuple

import networkx as nx
from networkx.algorithms.isomorphism import DiGraphMatcher

# ── local imports (service.analyze_ocel does the heavy lifting) ────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from mining.core import BehaviorGraph                           # noqa: E402
from service import (                                     # noqa: E402
    AnalysisResult, analyze_ocel, print_lattice_summary,
)


# ═══════════════════════════════════════════════════════════════════════════
# §1  BehaviorGraph  ➜  networkx.MultiDiGraph   (labeled)
# ═══════════════════════════════════════════════════════════════════════════

def _node_label(beh: BehaviorGraph, n: int) -> str:
    """Human-readable node label: activity + objects with cardinality."""
    act = beh.node_activity.get(n, "?")
    objs = beh.node_objects.get(n, frozenset())
    obj_part = ", ".join(
        f"{u}:{card}" for u, card in sorted(objs, key=lambda p: (str(p[0]), p[1]))
    ) or "∅"
    return f"{act} [{obj_part}]"


def _edge_label(beh: BehaviorGraph, ae: int) -> str:
    obj = beh.edge_label.get(ae, "?")
    ks, kt = beh.edge_kappa.get(ae, ("1", "1"))
    return f"{obj} ({ks}→{kt})"


def behavior_to_nx(beh: BehaviorGraph) -> nx.DiGraph:
    """
    Convert a BehaviorGraph to a labeled DiGraph.

    Parallel abstract-edges between the same (s, t) pair are folded into a
    single DiGraph edge whose ``labels`` attribute is a ``frozenset`` of the
    parallel edge labels.  This works because by construction abstract edges
    between a fixed (s, t) pair always have *distinct* labels (see
    ``compute_behavior``).
    """
    G: nx.DiGraph = nx.DiGraph()
    for n in beh.abstract_nodes:
        G.add_node(n, label=_node_label(beh, n))
    for ae in beh.abstract_edges:
        s, t = beh.edge_endpoints[ae]
        lbl = _edge_label(beh, ae)
        if G.has_edge(s, t):
            G[s][t]["labels"] = frozenset(G[s][t]["labels"] | {lbl})
        else:
            G.add_edge(s, t, labels=frozenset({lbl}))
    return G


# ═══════════════════════════════════════════════════════════════════════════
# §2  Canonical signature for labeled DiGraphs (small patterns)
# ═══════════════════════════════════════════════════════════════════════════

def canonical_signature(P: nx.DiGraph) -> Tuple:
    """
    Minimal-lex signature over all permutations of the node set.
    Only feasible for small patterns (≤ ~7 nodes ➜ ≤ 5040 perms).

    Signature: (tuple of node-labels, tuple of sorted edges), each edge
    being (src_idx, tgt_idx, sorted-tuple-of-labels).
    """
    nodes = list(P.nodes())
    node_labels = [P.nodes[n]["label"] for n in nodes]
    edges_raw = [
        (nodes.index(u), nodes.index(v), tuple(sorted(d["labels"])))
        for u, v, d in P.edges(data=True)
    ]

    n = len(nodes)
    best: Optional[Tuple] = None
    for perm in itertools.permutations(range(n)):
        inv = [0] * n
        for new_i, old_i in enumerate(perm):
            inv[old_i] = new_i
        nl = tuple(node_labels[perm[i]] for i in range(n))
        el = tuple(sorted(
            (inv[u], inv[v], lbls) for (u, v, lbls) in edges_raw
        ))
        sig = (nl, el)
        if best is None or sig < best:
            best = sig
    assert best is not None
    return best


def sig_to_graph(sig: Tuple) -> nx.DiGraph:
    """Rebuild a DiGraph from a canonical signature (used for rendering)."""
    node_labels, edges = sig
    G: nx.DiGraph = nx.DiGraph()
    for i, lbl in enumerate(node_labels):
        G.add_node(i, label=lbl)
    for (u, v, lbls) in edges:
        G.add_edge(u, v, labels=frozenset(lbls))
    return G


# ═══════════════════════════════════════════════════════════════════════════
# §3  SUBDUE-style labeled subgraph miner
# ═══════════════════════════════════════════════════════════════════════════

def _node_match(a: Dict, b: Dict) -> bool:
    return a.get("label") == b.get("label")


def _edge_match(a: Dict, b: Dict) -> bool:
    """Pattern (b) edge label-set must be a subset of host (a) edge label-set."""
    return b["labels"].issubset(a["labels"])


def _matcher(G: nx.DiGraph, P: nx.DiGraph) -> DiGraphMatcher:
    return DiGraphMatcher(G, P, node_match=_node_match, edge_match=_edge_match)


def support(P: nx.DiGraph, graphs: Sequence[nx.DiGraph]) -> int:
    """Number of graphs in which P has a subgraph isomorphism (once)."""
    cnt = 0
    for G in graphs:
        if _matcher(G, P).subgraph_is_isomorphic():
            cnt += 1
    return cnt


def _pattern_n_pseudo_edges(P: nx.DiGraph) -> int:
    """
    Treat each individual label in a label-set as one edge, since in the
    behavior graph parallel labels are genuinely distinct abstract edges.
    """
    return sum(len(d["labels"]) for _, _, d in P.edges(data=True))


def _seed_single_edge_patterns(
    graphs: Sequence[nx.DiGraph],
) -> Dict[Tuple, nx.DiGraph]:
    """All unique 2-node / 1-label-edge patterns across the dataset."""
    out: Dict[Tuple, nx.DiGraph] = {}
    for G in graphs:
        for u, v, d in G.edges(data=True):
            for lbl in d["labels"]:
                P = nx.DiGraph()
                P.add_node(0, label=G.nodes[u]["label"])
                P.add_node(1, label=G.nodes[v]["label"])
                P.add_edge(0, 1, labels=frozenset({lbl}))
                sig = canonical_signature(P)
                if sig not in out:
                    out[sig] = P
    return out


def _extend_pattern(
    P: nx.DiGraph,
    graphs: Sequence[nx.DiGraph],
) -> Dict[Tuple, nx.DiGraph]:
    """
    Enumerate all distinct one-label-edge extensions of ``P`` that actually
    occur in at least one host graph.  Deduplicated by canonical signature.
    """
    new_patterns: Dict[Tuple, nx.DiGraph] = {}

    for G in graphs:
        M = _matcher(G, P)
        for iso in M.subgraph_isomorphisms_iter():
            # networkx convention: iso maps G_node -> P_node
            G_to_P: Dict = iso
            mapped_G = set(G_to_P.keys())

            for u, v, d in G.edges(data=True):
                u_in, v_in = (u in mapped_G), (v in mapped_G)
                if not (u_in or v_in):
                    continue

                # Case A: edge between two already-mapped nodes → add a label
                # that is not yet in the pattern edge's label set.
                if u_in and v_in:
                    pu, pv = G_to_P[u], G_to_P[v]
                    current = (
                        P[pu][pv]["labels"] if P.has_edge(pu, pv) else frozenset()
                    )
                    for lbl in d["labels"] - current:
                        P2 = P.copy()
                        # frozenset is immutable, replace the whole attr
                        if P2.has_edge(pu, pv):
                            P2[pu][pv]["labels"] = frozenset(current | {lbl})
                        else:
                            P2.add_edge(pu, pv, labels=frozenset({lbl}))
                        sig = canonical_signature(P2)
                        new_patterns.setdefault(sig, P2)
                    continue

                # Case B: edge introduces a new node.  One label per extension.
                for lbl in d["labels"]:
                    P2 = P.copy()
                    new_id = max(P2.nodes()) + 1
                    if u_in:
                        P2.add_node(new_id, label=G.nodes[v]["label"])
                        P2.add_edge(G_to_P[u], new_id, labels=frozenset({lbl}))
                    else:
                        P2.add_node(new_id, label=G.nodes[u]["label"])
                        P2.add_edge(new_id, G_to_P[v], labels=frozenset({lbl}))
                    sig = canonical_signature(P2)
                    new_patterns.setdefault(sig, P2)

    return new_patterns


@dataclass
class MinedPattern:
    signature:  Tuple
    pattern:    nx.DiGraph
    support:    int
    size_edges: int
    size_nodes: int


def mine_subdue(
    graphs:       Sequence[nx.DiGraph],
    k:            int = 10,
    beam_width:   int = 12,
    max_edges:    int = 4,
    min_support:  int = 2,
    verbose:      bool = True,
) -> List[MinedPattern]:
    """
    SUBDUE-style beam search for frequent labeled subgraphs.

    The beam always holds the top ``beam_width`` patterns of the *current*
    size (by support).  Expansion: every pattern in the beam is extended by
    one edge (possibly introducing one new node).  All patterns ever seen
    with support ≥ ``min_support`` are collected and the top ``k`` are
    returned.
    """
    if not graphs:
        return []

    total_pseudo = sum(
        sum(len(d["labels"]) for _, _, d in g.edges(data=True))
        for g in graphs
    )
    if verbose:
        print(f"[mine] {len(graphs)} input graphs "
              f"(∑V={sum(g.number_of_nodes() for g in graphs)}, "
              f"∑E(labels)={total_pseudo})")

    # ── Level 1: single-edge seeds ─────────────────────────────────────────
    seeds = _seed_single_edge_patterns(graphs)
    if verbose:
        print(f"[mine] level 1: {len(seeds)} unique 1-edge patterns")

    scored: Dict[Tuple, MinedPattern] = {}
    for sig, P in seeds.items():
        sup = support(P, graphs)
        if sup >= min_support:
            scored[sig] = MinedPattern(sig, P, sup, 1, 2)

    beam = sorted(scored.values(), key=lambda m: -m.support)[:beam_width]

    # ── Beam expansion ─────────────────────────────────────────────────────
    for lvl in range(2, max_edges + 1):
        next_candidates: Dict[Tuple, nx.DiGraph] = {}
        for mp in beam:
            for sig, P2 in _extend_pattern(mp.pattern, graphs).items():
                next_candidates.setdefault(sig, P2)

        if verbose:
            print(f"[mine] level {lvl}: {len(next_candidates)} candidates")

        level_scored: List[MinedPattern] = []
        for sig, P in next_candidates.items():
            if sig in scored:
                continue
            sup = support(P, graphs)
            if sup < min_support:
                continue
            mp = MinedPattern(
                sig, P, sup,
                _pattern_n_pseudo_edges(P),
                P.number_of_nodes(),
            )
            scored[sig] = mp
            level_scored.append(mp)

        if not level_scored:
            break
        beam = sorted(level_scored, key=lambda m: -m.support)[:beam_width]

    top = sorted(
        scored.values(),
        key=lambda m: (-m.support, -m.size_edges, -m.size_nodes),
    )[:k]
    if verbose:
        print(f"[mine] done — {len(scored)} frequent patterns; top-{k} kept")
    return top


# ═══════════════════════════════════════════════════════════════════════════
# §4  Bottom / Top configuration helpers
# ═══════════════════════════════════════════════════════════════════════════

def bottom_config(result: AnalysisResult) -> Tuple[int, ...]:
    return tuple(0 for _ in result.types)


def top_config(result: AnalysisResult) -> Tuple[int, ...]:
    return tuple(result.hierarchies[tau].k for tau in result.types)


def behaviors_at(result: AnalysisResult, cfg: Tuple[int, ...]) -> List[BehaviorGraph]:
    cell = result.pyramid.get(cfg)
    if cell is None:
        raise KeyError(f"configuration {cfg} not in pyramid")
    return cell["behaviors"]


def config_to_string(result: AnalysisResult, cfg: Tuple[int, ...]) -> str:
    parts = []
    for i, tau in enumerate(result.types):
        name = result.level_names.get(tau, [])
        v = cfg[i]
        lbl = name[v] if 0 <= v < len(name) else str(v)
        parts.append(f"{tau}={lbl}")
    return "γ(" + ", ".join(parts) + ")"


# ═══════════════════════════════════════════════════════════════════════════
# §5  HTML report (Cytoscape.js, self-contained)
# ═══════════════════════════════════════════════════════════════════════════

def _pattern_to_cy_elements(P: nx.DiGraph) -> List[Dict]:
    """
    Cytoscape elements for the pattern.  A pattern edge may carry multiple
    labels (frozenset); render one Cytoscape edge *per* label so parallel
    abstract-edges appear visually distinct.
    """
    nodes = [
        {"data": {"id": f"n{n}", "label": d["label"]}}
        for n, d in P.nodes(data=True)
    ]
    edges: List[Dict] = []
    eid = 0
    for u, v, d in P.edges(data=True):
        for lbl in sorted(d["labels"]):
            edges.append({"data": {
                "id": f"e{eid}",
                "source": f"n{u}",
                "target": f"n{v}",
                "label": lbl,
            }})
            eid += 1
    return nodes + edges


_HTML_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>SUBDUE — top-{{ k }} frequent patterns over {{ columns|length }} lattice node(s)</title>
<script src="https://unpkg.com/cytoscape@3.28.1/dist/cytoscape.min.js"></script>
<script src="https://unpkg.com/dagre@0.8.5/dist/dagre.min.js"></script>
<script src="https://unpkg.com/cytoscape-dagre@2.5.0/cytoscape-dagre.js"></script>
<style>
  :root {
    --bg: #f7f7fb;
    --card: #fff;
    --ink: #1f2937;
    --muted: #6b7280;
    --border: #e5e7eb;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto,
                 "Helvetica Neue", Arial, sans-serif;
    background: var(--bg);
    color: var(--ink);
  }
  header {
    padding: 20px 28px;
    background: #fff;
    border-bottom: 1px solid var(--border);
    position: sticky;
    top: 0;
    z-index: 5;
  }
  header h1 { margin: 0 0 6px 0; font-size: 18px; font-weight: 600; }
  header .meta { color: var(--muted); font-size: 13px; }
  header code { background: #f1f1f5; padding: 1px 6px; border-radius: 4px; }

  .scroll-hint {
    color: var(--muted);
    font-size: 12px;
    padding: 10px 28px 0;
  }

  main {
    display: flex;
    flex-direction: row;
    gap: 18px;
    padding: 14px 28px 32px;
    overflow-x: auto;
    scroll-snap-type: x proximity;
  }
  main::-webkit-scrollbar { height: 10px; }
  main::-webkit-scrollbar-thumb {
    background: #d1d5db;
    border-radius: 5px;
  }

  .col {
    flex: 0 0 520px;
    display: flex;
    flex-direction: column;
    gap: 14px;
    scroll-snap-align: start;
  }
  .col-head {
    font-size: 12px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    padding: 6px 10px;
    border-radius: 6px;
    color: #fff;
    width: fit-content;
  }
  .col .cfg {
    font-family: monospace;
    font-size: 12px;
    color: var(--ink);
    background: #fff;
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 6px 8px;
    word-break: break-word;
  }
  .col .desc { color: var(--muted); font-size: 12px; }

  .pattern-card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 12px 14px;
    box-shadow: 0 1px 3px rgba(0,0,0,.04);
  }
  .pattern-card h3 {
    margin: 0 0 4px 0;
    font-size: 14px;
    font-weight: 600;
    display: flex;
    justify-content: space-between;
    align-items: baseline;
  }
  .pattern-card h3 .rank {
    color: var(--muted);
    font-weight: 500;
    font-size: 12px;
  }
  .metrics {
    display: flex;
    gap: 14px;
    color: var(--muted);
    font-size: 12px;
    margin-bottom: 8px;
    flex-wrap: wrap;
  }
  .metrics b { color: var(--ink); font-weight: 600; }
  .cy {
    width: 100%;
    height: 340px;
    background: #fafbff;
    border: 1px dashed var(--border);
    border-radius: 6px;
  }
  .empty {
    color: var(--muted);
    font-style: italic;
    padding: 24px;
    text-align: center;
    border: 1px dashed var(--border);
    border-radius: 8px;
    background: var(--card);
  }
</style>
</head>
<body>
<header>
  <h1>SUBDUE — labeled subgraph mining on OCEL behaviors</h1>
  <div class="meta">
    OCEL: <code>{{ ocel_path }}</code>
    &nbsp;·&nbsp; mode = <b>{{ mode }}</b>
    &nbsp;·&nbsp; lattice nodes = <b>{{ columns|length }}</b>
    &nbsp;·&nbsp; k = <b>{{ k }}</b>
    &nbsp;·&nbsp; max edges = <b>{{ max_edges }}</b>
    &nbsp;·&nbsp; beam = <b>{{ beam_width }}</b>
    &nbsp;·&nbsp; min support = <b>{{ min_support }}</b>
    &nbsp;·&nbsp; elapsed = <b>{{ elapsed_s }} s</b>
  </div>
</header>

{% if columns|length > 2 %}
<div class="scroll-hint">← scroll horizontally to compare lattice nodes →</div>
{% endif %}

<main>
  {% for col in columns %}
  <section class="col" id="col-{{ loop.index0 }}">
    <div class="col-head" style="background: {{ col.accent }}">{{ col.title }}</div>
    <div class="cfg">{{ col.cfg_str }}</div>
    <div class="desc">{{ col.n_graphs }} behavior graph(s)</div>

    {% if col.patterns %}
      {% for p in col.patterns %}
      <div class="pattern-card">
        <h3>
          <span>Pattern {{ loop.index }}</span>
          <span class="rank">support {{ p.support }} / {{ col.n_graphs }}</span>
        </h3>
        <div class="metrics">
          <span><b>{{ p.size_nodes }}</b> nodes</span>
          <span><b>{{ p.size_edges }}</b> edges</span>
          {% if col.n_graphs %}
          <span>coverage <b>{{ (100 * p.support / col.n_graphs) | round(1) }}%</b></span>
          {% endif %}
        </div>
        <div class="cy" id="cy-{{ loop.index0 }}-c{{ col.idx }}"></div>
      </div>
      {% endfor %}
    {% else %}
      <div class="empty">No patterns with support ≥ {{ min_support }}.</div>
    {% endif %}
  </section>
  {% endfor %}
</main>

<script>
  // COLUMNS: array of { accent, patterns: [elements, ...] }
  const COLUMNS = {{ columns_json | safe }};

  const baseStyle = [
    { selector: "node", style: {
      "background-color": "#4E79A7",
      "label": "data(label)",
      "color": "#fff",
      "font-size": "11px",
      "font-family": "monospace",
      "text-wrap": "wrap",
      "text-max-width": "170px",
      "text-valign": "center",
      "text-halign": "center",
      "width": "170px",
      "height": "58px",
      "shape": "roundrectangle",
      "text-outline-width": 0,
    }},
    { selector: "edge", style: {
      "curve-style": "bezier",
      "control-point-step-size": 40,
      "target-arrow-shape": "triangle",
      "target-arrow-color": "#6b7280",
      "line-color": "#6b7280",
      "width": 2,
      "label": "data(label)",
      "font-size": "10px",
      "font-family": "monospace",
      "color": "#374151",
      "text-background-color": "#fff",
      "text-background-opacity": 0.9,
      "text-background-padding": "3px",
      "text-rotation": "autorotate",
    }},
  ];

  function renderColumn(colIdx, col) {
    col.patterns.forEach((elems, pi) => {
      const ss = baseStyle.map(s => ({...s, style: {...s.style}}));
      ss[0].style["background-color"] = col.accent;
      const el = document.getElementById(`cy-${pi}-c${colIdx}`);
      if (!el) return;
      cytoscape({
        container: el,
        elements: elems,
        style: ss,
        layout: { name: "dagre", rankDir: "LR", nodeSep: 28, rankSep: 60 },
        userZoomingEnabled: true,
        userPanningEnabled: true,
        boxSelectionEnabled: false,
      });
    });
  }

  COLUMNS.forEach((col, i) => renderColumn(i, col));
</script>
</body>
</html>
"""


@dataclass
class ColumnData:
    """One rendered column in the HTML report."""
    idx:       int
    title:     str                    # e.g. "Bottom", "Top", "Node 3"
    cfg_str:   str                    # e.g. "γ(Order=id, Item=type)"
    accent:    str                    # hex colour
    n_graphs:  int
    patterns:  List[MinedPattern]


def _lerp_hex(a: str, b: str, t: float) -> str:
    """Linear interpolate two #rrggbb colours, t in [0, 1]."""
    ax = int(a[1:3], 16), int(a[3:5], 16), int(a[5:7], 16)
    bx = int(b[1:3], 16), int(b[3:5], 16), int(b[5:7], 16)
    r, g, bl = (round((1 - t) * ax[i] + t * bx[i]) for i in range(3))
    return f"#{r:02X}{g:02X}{bl:02X}"


def column_accent(level: int, max_level: int) -> str:
    """Orange (low) → blue (high), same palette as the main GUI."""
    if max_level <= 0:
        return "#4E79A7"
    t = level / max_level
    return _lerp_hex("#F28E2B", "#4E79A7", t)


def render_html(
    *,
    ocel_path:    str,
    mode:         str,
    columns:      List[ColumnData],
    k:            int,
    max_edges:    int,
    beam_width:   int,
    min_support:  int,
    elapsed_s:    float,
    out_path:     str,
) -> None:
    from jinja2 import Template

    columns_for_js = [
        {
            "accent":   col.accent,
            "patterns": [_pattern_to_cy_elements(p.pattern) for p in col.patterns],
        }
        for col in columns
    ]

    html_text = Template(_HTML_TEMPLATE).render(
        ocel_path    = html.escape(ocel_path),
        mode         = mode,
        columns      = columns,
        columns_json = json.dumps(columns_for_js),
        k            = k,
        max_edges    = max_edges,
        beam_width   = beam_width,
        min_support  = min_support,
        elapsed_s    = f"{elapsed_s:.1f}",
    )
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(html_text)


# ═══════════════════════════════════════════════════════════════════════════
# §6  CLI
# ═══════════════════════════════════════════════════════════════════════════

def _config_height(result: AnalysisResult, cfg: Tuple[int, ...]) -> int:
    """Sum of per-type heights — used both for sorting and for the colour lerp."""
    return sum(
        result.hierarchies[tau].height(cfg[i])
        for i, tau in enumerate(result.types)
    )


def _configs_to_mine(result: AnalysisResult, mode: str) -> List[Tuple[int, ...]]:
    """Return the list of lattice configs to mine, sorted bottom→top."""
    if mode == "bottom-top":
        return [bottom_config(result), top_config(result)]
    if mode == "all":
        return sorted(
            result.pyramid.keys(),
            key=lambda c: (_config_height(result, c), c),
        )
    raise ValueError(f"unknown mode: {mode!r}")


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description="SUBDUE-style subgraph mining on OCEL behavior graphs. "
                    "Runs on the bottom + top lattice nodes by default, "
                    "or on every lattice node with --mode all.",
    )
    p.add_argument("ocel", help="Path to OCEL (.sqlite / .json / .xml / .jsonocel)")
    p.add_argument("--hierarchy", default=None,
                   help="Optional hierarchy JSON (see service.load_hierarchy_from_json).")
    p.add_argument("--mode", choices=("bottom-top", "all"), default="bottom-top",
                   help="'bottom-top' (default) mines only the bottom and top "
                        "configurations; 'all' mines every lattice node.")
    p.add_argument("-k", "--k", type=int, default=10,
                   help="Keep top K patterns per lattice node (default: 10).")
    p.add_argument("--max-edges", type=int, default=4,
                   help="Maximum pattern size in edges (default: 4).")
    p.add_argument("--beam", type=int, default=12,
                   help="Beam width for expansion (default: 12).")
    p.add_argument("--min-support", type=int, default=2,
                   help="Minimum number of graphs a pattern must appear in (default: 2).")
    p.add_argument("--out", default="subdue_report.html",
                   help="Output HTML path (default: subdue_report.html).")
    p.add_argument("--quiet", action="store_true", help="Silence progress logs.")
    args = p.parse_args(argv)

    t0 = time.time()
    if not args.quiet:
        print(f"[load] analyzing {args.ocel} ...")
    result = analyze_ocel(args.ocel, args.hierarchy)

    # ── lattice summary & bottom/top metrics (always printed) ──────────────
    if not args.quiet:
        print_lattice_summary(result)

    # ── configs to mine ────────────────────────────────────────────────────
    configs = _configs_to_mine(result, args.mode)
    b_cfg = bottom_config(result)
    t_cfg = top_config(result)
    max_height = max(_config_height(result, c) for c in configs) or 1

    if not args.quiet:
        print(f"[mode] {args.mode}  →  {len(configs)} lattice node(s) to mine")

    # ── mine each config, build columns ────────────────────────────────────
    columns: List[ColumnData] = []
    for i, cfg in enumerate(configs):
        behs = behaviors_at(result, cfg)
        graphs = [behavior_to_nx(b) for b in behs if b.abstract_nodes]

        # Column title & colour
        if cfg == b_cfg and cfg == t_cfg:
            title = "Node"           # degenerate 1-node lattice
        elif cfg == b_cfg:
            title = "Bottom"
        elif cfg == t_cfg:
            title = "Top"
        else:
            title = f"Node {i}"

        h = _config_height(result, cfg)
        accent = column_accent(h, max_height)

        if not args.quiet:
            print(f"[mine] column {i+1}/{len(configs)} "
                  f"({title}, h={h}): {len(graphs)} non-empty graphs")

        if graphs:
            patterns = mine_subdue(
                graphs, k=args.k, beam_width=args.beam,
                max_edges=args.max_edges, min_support=args.min_support,
                verbose=not args.quiet,
            )
        else:
            patterns = []

        columns.append(ColumnData(
            idx       = i,
            title     = title,
            cfg_str   = config_to_string(result, cfg),
            accent    = accent,
            n_graphs  = len(graphs),
            patterns  = patterns,
        ))

    elapsed = time.time() - t0
    render_html(
        ocel_path    = args.ocel,
        mode         = args.mode,
        columns      = columns,
        k            = args.k,
        max_edges    = args.max_edges,
        beam_width   = args.beam,
        min_support  = args.min_support,
        elapsed_s    = elapsed,
        out_path     = args.out,
    )
    if not args.quiet:
        print(f"[done] {elapsed:.1f}s — wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
