#!/usr/bin/env python3
from __future__ import annotations

import io
import itertools
import os
import sys
import tempfile
from contextlib import redirect_stdout
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import networkx as nx
from networkx.algorithms.isomorphism import DiGraphMatcher

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)



# ── definitions extracted from subdue_experiment (no subdue dependency) ──
@dataclass
class MinedPattern:
    signature:  Tuple
    pattern:    nx.DiGraph
    support:    int
    size_edges: int
    size_nodes: int

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


def _node_match(a, b):
    return a.get("label") == b.get("label")

def _edge_match(a, b):
    return b["labels"].issubset(a["labels"])

def _matcher(G, P):
    return DiGraphMatcher(G, P, node_match=_node_match, edge_match=_edge_match)

def support(P: nx.DiGraph, graphs: Sequence[nx.DiGraph]) -> int:
    """Number of graphs in which P has a subgraph isomorphism (once)."""
    cnt = 0
    for G in graphs:
        if _matcher(G, P).subgraph_is_isomorphic():
            cnt += 1
    return cnt
# ─────────────────────────────────────────────────────────────────────────

_GADGET_PREFIX = "__edge_"
_IN_LABEL = "_in"
_OUT_LABEL = "_out"


def _encode_graph(G: nx.DiGraph) -> nx.DiGraph:
    H: nx.DiGraph = nx.DiGraph()
    id_of: Dict[Any, int] = {}
    ctr = itertools.count()

    for n, d in G.nodes(data=True):
        nid = next(ctr)
        id_of[n] = nid
        H.add_node(nid, label=d.get("label", "?"))

    for u, v, d in G.edges(data=True):
        u_id = id_of[u]
        v_id = id_of[v]
        for lbl in d.get("labels", frozenset()):
            m = next(ctr)
            H.add_node(m, label=f"{_GADGET_PREFIX}{lbl}")
            H.add_edge(u_id, m, label=_IN_LABEL)
            H.add_edge(m, v_id, label=_OUT_LABEL)
    return H

def _pattern_n_pseudo_edges(P: nx.DiGraph) -> int:
    return sum(len(d["labels"]) for _, _, d in P.edges(data=True))

def _build_vocabs(
    encoded: Sequence[nx.DiGraph],
) -> Tuple[Dict[str, int], Dict[str, int], Dict[int, str], Dict[int, str]]:
    nv: Dict[str, int] = {}
    ev: Dict[str, int] = {}
    for G in encoded:
        for _, d in G.nodes(data=True):
            lbl = d.get("label", "?")
            if lbl not in nv:
                nv[lbl] = len(nv)
        for _, _, d in G.edges(data=True):
            lbl = d.get("label", "?")
            if lbl not in ev:
                ev[lbl] = len(ev)
    return nv, ev, {v: k for k, v in nv.items()}, {v: k for k, v in ev.items()}

def _write_gspan_file(
    encoded: Sequence[nx.DiGraph],
    path: str,
    node_vocab: Dict[str, int],
    edge_vocab: Dict[str, int],
) -> None:
    with open(path, "w") as f:
        for gi, G in enumerate(encoded):
            f.write(f"t # {gi}\n")
            for n, d in G.nodes(data=True):
                f.write(f"v {n} {node_vocab[d['label']]}\n")
            for u, v, d in G.edges(data=True):
                f.write(f"e {u} {v} {edge_vocab[d['label']]}\n")
        f.write("t # -1\n")

def _import_gspan():
    try:
        from gspan_mining.gspan import gSpan    # type: ignore
    except ImportError as exn:
        raise ImportError(
            "gspan-mining is not installed. Install with:\n"
            "    pip install gspan-mining"
        ) from exn
    return gSpan


def _gspan_graph_to_nx(g_obj, inv_nv: Dict[int, str],
                       inv_ev: Dict[int, str]) -> Optional[nx.DiGraph]:
    if hasattr(g_obj, "to_graph") and not hasattr(g_obj, "vertices"):
        try:
            g_obj = g_obj.to_graph()
        except TypeError:
            try:
                g_obj = g_obj.to_graph(0)
            except Exception:
                return None

    verts = getattr(g_obj, "vertices", None)
    if verts is None:
        return None
    P: nx.DiGraph = nx.DiGraph()
    for vid, vx in verts.items():
        lbl_int = getattr(vx, "vlb", None)
        if lbl_int is None:
            return None
        P.add_node(int(vid), label=inv_nv.get(int(lbl_int), "?"))
    for vid, vx in verts.items():
        edges = getattr(vx, "edges", {}) or {}
        iter_edges = edges.values() if hasattr(edges, "values") else edges
        for edge in iter_edges:
            to = getattr(edge, "to",  None)
            elbl = getattr(edge, "elb", None)
            if to is None or elbl is None:
                continue
            P.add_edge(int(vid), int(to),
                       label=inv_ev.get(int(elbl), "?"))
    return P


def _harvest_frequent_subgraphs(gs) -> List[Any]:
    for attr in ("_frequent_subgraphs", "frequent_subgraphs",
                 "_frequent_graphs"):
        obj = getattr(gs, attr, None)
        if obj is None:
            continue
        if isinstance(obj, list):
            return obj
        if isinstance(obj, dict):
            return list(obj.values())
    return []


def _parse_gspan_text_output(
    text: str,
    inv_nv: Dict[int, str],
    inv_ev: Dict[int, str],
) -> List[nx.DiGraph]:
    out: List[nx.DiGraph] = []
    cur: Optional[nx.DiGraph] = None
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("t"):
            if cur is not None and cur.number_of_nodes() > 0:
                out.append(cur)
            cur = nx.DiGraph()
            continue
        if cur is None:
            continue
        parts = line.split()
        try:
            if parts[0] == "v" and len(parts) >= 3:
                cur.add_node(int(parts[1]),
                             label=inv_nv.get(int(parts[2]), "?"))
            elif parts[0] == "e" and len(parts) >= 4:
                cur.add_edge(int(parts[1]), int(parts[2]),
                             label=inv_ev.get(int(parts[3]), "?"))
        except ValueError:
            continue
    if cur is not None and cur.number_of_nodes() > 0:
        out.append(cur)
    return out

def _decode_pattern(Pe: nx.DiGraph) -> Optional[nx.DiGraph]:
    Q: nx.DiGraph = nx.DiGraph()
    id_map: Dict[int, int] = {}
    ctr = itertools.count()

    for n, d in Pe.nodes(data=True):
        lbl = d.get("label", "")
        if isinstance(lbl, str) and lbl.startswith(_GADGET_PREFIX):
            continue
        qid = next(ctr)
        id_map[n] = qid
        Q.add_node(qid, label=lbl)

    for n, d in Pe.nodes(data=True):
        lbl = d.get("label", "")
        if not (isinstance(lbl, str) and lbl.startswith(_GADGET_PREFIX)):
            continue

        src = None
        dst = None

        for u, _, ed in Pe.in_edges(n, data=True):
            if ed.get("label") == _IN_LABEL:
                src = u

        for _, v, ed in Pe.out_edges(n, data=True):
            if ed.get("label") == _OUT_LABEL:
                dst = v

        if src is None or dst is None:
            return None

        if src not in id_map or dst not in id_map:
            return None

        edge_label = lbl[len(_GADGET_PREFIX):]
        qsrc = id_map[src]
        qdst = id_map[dst]

        if Q.has_edge(qsrc, qdst):
            Q[qsrc][qdst]["labels"] = frozenset(
                set(Q[qsrc][qdst]["labels"]) | {edge_label}
            )
        else:
            Q.add_edge(qsrc, qdst, labels=frozenset({edge_label}))

    if Q.number_of_nodes() < 2 or Q.number_of_edges() == 0:
        return None

    if not nx.is_weakly_connected(Q):
        return None

    return Q

def mine_gspan(
    graphs: Sequence[nx.DiGraph],
    k: int = 10,
    max_edges: int = 4,
    min_support: int = 2,
    verbose: bool = True,
    **_ignored,
) -> List[MinedPattern]:
    if not graphs:
        return []

    gSpan = _import_gspan()

    if verbose:
        n_v = sum(g.number_of_nodes() for g in graphs)
        n_e = sum(_pattern_n_pseudo_edges(g) for g in graphs)
        print(f"[gspan] {len(graphs)} input graphs "
              f"(∑V={n_v}, ∑E(labels)={n_e})")

    encoded = [_encode_graph(G) for G in graphs]
    nv, ev, inv_nv, inv_ev = _build_vocabs(encoded)

    if verbose:
        e_v = sum(g.number_of_nodes() for g in encoded)
        e_e = sum(g.number_of_edges() for g in encoded)
        print(f"[gspan] encoded: ∑V={e_v}, ∑E={e_e}, "
              f"|V-vocab|={len(nv)}, |E-vocab|={len(ev)}")

    max_enc_vertices = 2 * max_edges + 2

    with tempfile.TemporaryDirectory() as tmpdir:
        data_path = os.path.join(tmpdir, "graphs.data")
        _write_gspan_file(encoded, data_path, nv, ev)

        captured = io.StringIO()
        class _Tee:
            def __init__(self, *streams): self._streams = streams
            def write(self, x):
                for s in self._streams:
                    s.write(x)
                return len(x)
            def flush(self):
                for s in self._streams:
                    s.flush()
        target = _Tee(captured, sys.stdout) if verbose else captured
        ctx = redirect_stdout(target)

        kwargs = dict(
            database_file_name = data_path,
            min_support = min_support,
            min_num_vertices = 3,
            max_num_vertices = max_enc_vertices,
            is_undirected = False,
            where=False,
            verbose = False
        )
        with ctx:
            try:
                gs = gSpan(**kwargs, max_ngraphs=float("inf"),
                           visualize=False)
            except TypeError:
                try:
                    gs = gSpan(**kwargs)
                except TypeError:
                    gs = gSpan(
                        database_file_name = data_path,
                        min_support        = min_support,
                        min_num_vertices   = 3,
                        max_num_vertices   = max_enc_vertices,
                    )
            gs.run()
            reporter = getattr(gs, "report", None) or getattr(gs, "_report", None)
            if callable(reporter):
                try:
                    reporter()
                except Exception:
                    pass
        gs_stdout_text = captured.getvalue()

    freq = _harvest_frequent_subgraphs(gs)


    source = "attribute"
    if not freq:
        freq = _parse_gspan_text_output(gs_stdout_text, inv_nv, inv_ev)
        source = "stdout-parse"
    if verbose:
        print(f"[gspan] harvested {len(freq)} raw frequent subgraphs "
              f"(source: {source})")

    scored: Dict[Tuple, MinedPattern] = {}
    n_raw = 0
    n_pe_none = 0
    n_decode_none = 0
    n_too_large = 0
    n_support_low = 0
    n_duplicate = 0
    n_kept = 0

    for idx, gsp_g in enumerate(freq, 1):
        n_raw += 1

        if idx % 500 == 0:
            print(
                f"[gspan verify] {idx}/{len(freq)} checked | "
                f"decoded={n_raw - n_decode_none - n_pe_none} "
                f"kept={n_kept}"
            )

        Pe = gsp_g if isinstance(gsp_g, nx.DiGraph) \
            else _gspan_graph_to_nx(gsp_g, inv_nv, inv_ev)

        if Pe is None:
            n_pe_none += 1
            continue

        P = _decode_pattern(Pe)
        if P is None:
            n_decode_none += 1
            continue

        n_edges_orig = _pattern_n_pseudo_edges(P)
        if n_edges_orig == 0 or n_edges_orig > max_edges:
            n_too_large += 1
            continue

        sig = canonical_signature(P)
        if sig in scored:
            n_duplicate += 1
            continue

        sup = support(P, graphs)
        if sup < min_support:
            n_support_low += 1
            continue

        n_kept += 1
        scored[sig] = MinedPattern(
            signature=sig,
            pattern=P,
            support=sup,
            size_edges=n_edges_orig,
            size_nodes=P.number_of_nodes(),
        )

    print("[debug gspan filter]")
    print("raw:", n_raw)
    print("pe_none:", n_pe_none)
    print("decode_none:", n_decode_none)
    print("too_large:", n_too_large)
    print("support_low:", n_support_low)
    print("duplicate:", n_duplicate)
    print("kept:", n_kept)

    if verbose:
        print(f"[gspan] {len(scored)} decoded+verified frequent patterns "
              f"(after gadget-decode & support recheck)")

    top = sorted(
        scored.values(),
        key=lambda m: (-m.support, -m.size_edges, -m.size_nodes),
    )[:k]
    return top

def mine(
    miner:       str,
    graphs:      Sequence[nx.DiGraph],
    *,
    k:           int,
    max_edges:   int,
    min_support: int,
    beam_width:  int  = 12,
    verbose:     bool = True,
) -> List[MinedPattern]:
    return mine_gspan(
            graphs,
            k           = k,
            max_edges   = max_edges,
            min_support = min_support,
            verbose     = verbose,
        )

def _smoke_test(verbose: bool = True) -> None:
    def _g(nodes, edges):
        g = nx.DiGraph()
        for n, lbl in nodes:
            g.add_node(n, label=lbl)
        for u, v, lbls in edges:
            g.add_edge(u, v, labels=frozenset(lbls))
        return g

    motif_graphs = [
        _g([(0, "A"), (1, "B"), (2, "C")],
           [(0, 1, {"o1"}), (1, 2, {"o1"})]),
        _g([(0, "A"), (1, "B"), (2, "C"), (3, "X")],
           [(0, 1, {"o1"}), (1, 2, {"o1"}), (1, 3, {"o2"})]),
        _g([(0, "A"), (1, "B"), (2, "C")],
           [(0, 1, {"o1", "o2"}), (1, 2, {"o1"})]),
        _g([(0, "A"), (1, "B"), (2, "C")],
           [(0, 1, {"o1"}), (1, 2, {"o1", "o3"})]),
    ]
    noise = _g([(0, "P"), (1, "Q")], [(0, 1, {"z"})])
    corpus = motif_graphs + [noise]

    patterns = mine_gspan(
        corpus, k=20, max_edges=3, min_support=2, verbose=verbose,
    )
    if verbose:
        print(f"\n[smoke] {len(patterns)} patterns mined from {len(corpus)} toys:")
        for mp in patterns:
            edges = [(u, v, sorted(d['labels']))
                     for u, v, d in mp.pattern.edges(data=True)]
            nodes = [(n, mp.pattern.nodes[n]['label'])
                     for n in mp.pattern.nodes()]
            print(f"  support={mp.support}  |V|={mp.size_nodes} "
                  f"|E|={mp.size_edges}   nodes={nodes}  edges={edges}")

    assert any(mp.support >= 4 for mp in patterns), \
        "smoke test failed: no pattern found with support >= 4"
    if verbose:
        print("[smoke] OK\n")


if __name__ == "__main__":
    _smoke_test(verbose=True)
