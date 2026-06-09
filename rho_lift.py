#!/usr/bin/env python3
from __future__ import annotations
import argparse
import csv
import datetime as _dt
import html
import itertools
import json
import os
import pickle
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean
from typing import Any, Callable, Dict, FrozenSet, List, Optional, Sequence, Set, Tuple
import networkx as nx
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
from mining.core import BehaviorGraph, OBJ_ID, LevelConfiguration, OCELLevelGraph, ObjectTypeHierarchy, ProcessExecution, compute_behavior, compute_iso_classes
from mining.mining_gspan import mine as mine_dispatch
from mining.kpi import KPI, BUILTIN_KPIS, compute_kpi_matrix, resolve_kpi_flags
import pm4py
from networkx.algorithms.isomorphism import DiGraphMatcher

def _node_match(a: Dict, b: Dict) -> bool:
    return a.get("label") == b.get("label")

def _edge_match(a: Dict, b: Dict) -> bool:
    return b["labels"].issubset(a["labels"])

def _matcher(G: nx.DiGraph, P: nx.DiGraph) -> DiGraphMatcher:
    return DiGraphMatcher(G, P, node_match=_node_match, edge_match=_edge_match)

def _lerp_hex(a: str, b: str, t: float) -> str:
    ax = int(a[1:3], 16), int(a[3:5], 16), int(a[5:7], 16)
    bx = int(b[1:3], 16), int(b[3:5], 16), int(b[5:7], 16)
    r, g, bl = (round((1 - t) * ax[i] + t * bx[i]) for i in range(3))
    return f"#{r:02X}{g:02X}{bl:02X}"

def column_accent(level: int, max_level: int) -> str:
    if max_level <= 0:
        return "#4E79A7"
    t = level / max_level
    return _lerp_hex("#F28E2B", "#4E79A7", t)

def canonical_signature(P: nx.DiGraph) -> Tuple:
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

def _node_label(beh: BehaviorGraph, n: int) -> str:
    act = beh.node_activity.get(n, "?")
    objs = beh.node_objects.get(n, frozenset())
    obj_part = ", ".join(
        f"{u}:{card}" for u, card in sorted(objs, key=lambda p: (str(p[0]), p[1]))
    ) or "emptyset"
    return f"{act} [{obj_part}]"


def _edge_label(beh: BehaviorGraph, ae: int) -> str:
    obj = beh.edge_label.get(ae, "?")
    ks, kt = beh.edge_kappa.get(ae, ("1", "1"))
    return f"{obj} ({ks}->{kt})"

def behavior_to_nx(beh: BehaviorGraph) -> nx.DiGraph:
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

def _load_ocel_sqlite_pm4py(path: str):
    return pm4py.read_ocel2_sqlite(path)

def _build_process_executions_from_ocpa(ocel, leading_type: str, sqlite_path: str, verbose: bool=True) -> List[ProcessExecution]:
    import pandas as pd
    import numpy as np
    try:
        from ocpa.objects.log.importer.ocel2.sqlite import factory as imp
    except ImportError as ex:
        raise RuntimeError('ocpa is required for leading-type execution extraction — install it with `pip install ocpa`.') from ex
    params = {'execution_extraction': 'leading_type', 'leading_type': leading_type}
    ocpa_ocel = imp.apply(sqlite_path, parameters=params)
    log_df = ocpa_ocel.log.log if hasattr(ocpa_ocel.log, 'log') else ocpa_ocel.log
    EID = next((c for c in ('event_id', 'ocel:eid', 'ocel_id', 'eid') if c in log_df.columns), None)
    ACT = next((c for c in ('event_activity', 'ocel:activity', 'activity') if c in log_df.columns), None)
    TS = next((c for c in ('event_timestamp', 'ocel:timestamp', 'timestamp') if c in log_df.columns), None)
    if EID is None or ACT is None or TS is None:
        raise RuntimeError(f'Could not locate ocpa event columns in log_df (columns: {list(log_df.columns)}).')
    std_cols = {EID, ACT, TS}
    obj_types: List[str] = []
    if hasattr(ocpa_ocel, 'object_types'):
        obj_types = [t for t in ocpa_ocel.object_types if t in log_df.columns and t not in std_cols]
    if not obj_types:
        for c in log_df.columns:
            if c in std_cols:
                continue
            for i in range(min(len(log_df), 20)):
                sample = log_df[c].iloc[i]
                if isinstance(sample, (list, set, tuple, np.ndarray)):
                    obj_types.append(c)
                    break

    def _iter_ids(cell):
        if cell is None:
            return
        try:
            if isinstance(cell, float) and np.isnan(cell):
                return
        except Exception:
            pass
        if isinstance(cell, str):
            yield cell
            return
        try:
            for x in cell:
                if x is None:
                    continue
                yield x
        except TypeError:
            yield cell
    act_map: Dict[str, str] = {}
    ts_map: Dict[str, Any] = {}
    e2o: Dict[str, Set[str]] = defaultdict(set)
    o2events: Dict[str, List[Tuple[str, Any]]] = defaultdict(list)
    for _, row in log_df.iterrows():
        eid = str(row[EID])
        ts = row[TS]
        act_map[eid] = row[ACT]
        ts_map[eid] = ts
        for t in obj_types:
            for oid in _iter_ids(row[t]):
                oid_s = str(oid)
                e2o[eid].add(oid_s)
                o2events[oid_s].append((eid, ts))
    o2trace_global: Dict[str, List[str]] = {}
    for oid, pairs in o2events.items():
        pairs.sort(key=lambda p: p[1])
        o2trace_global[oid] = [p[0] for p in pairs]
    if verbose:
        print(f' log_df: {len(log_df)} events, obj-type cols: {obj_types}')
        print(f' indices: {len(act_map)} events, {len(o2trace_global)} objects tracked.')
        if o2trace_global:
            sample_oids = list(o2trace_global.keys())[:3]
            for oid in sample_oids:
                print(f' sample obj {oid!r}: trace len = {len(o2trace_global[oid])}')
    raw_pes = list(ocpa_ocel.process_executions)
    out: List[ProcessExecution] = []
    diag_edges = 0
    diag_events = 0
    diag_objects = 0
    diag_empty_objects = 0
    sample_printed = 0
    for ev_idx_set in raw_pes:
        evs: List[str] = []
        for idx in ev_idx_set:
            if isinstance(idx, (int, np.integer)):
                evs.append(str(log_df[EID].iloc[int(idx)]))
            else:
                evs.append(str(idx))
        E_X: Set[str] = set(evs)
        X_set: Set[str] = set()
        for e in E_X:
            X_set.update(e2o.get(e, ()))
        X: FrozenSet = frozenset(X_set)
        if not X:
            diag_empty_objects += 1
        D_X: List[Tuple[str, str, str]] = []
        for oid in X:
            trace = [e for e in o2trace_global.get(oid, ()) if e in E_X]
            for i in range(len(trace) - 1):
                D_X.append((trace[i], trace[i + 1], oid))
        einfo: Dict[str, Dict] = {e: {'activity': act_map.get(e), 'timestamp': ts_map.get(e), 'objects': e2o.get(e, set()) & X} for e in E_X}
        pe = ProcessExecution(objects=X, events=E_X, edges=D_X, event_info=einfo)
        out.append(pe)
        diag_edges += len(D_X)
        diag_events += len(E_X)
        diag_objects += len(X)
        if verbose and sample_printed < 3:
            print(f' sample exec[{sample_printed}]: |E|={len(E_X)}  |O|={len(X)}  |D|={len(D_X)}')
            sample_printed += 1
    if verbose:
        n = max(1, len(out))
        print(f'[exec] edge reconstruction: {diag_edges} edges across {len(out)} executions ({diag_events} events, {diag_objects} object-refs; avg {diag_edges / n:.1f} edges/exec, {diag_objects / n:.1f} objs/exec' + (f', {diag_empty_objects} empty-object execs' if diag_empty_objects else '') + ').')
    return out
_STD_OBJ_COLS = {OBJ_ID, 'ocel:type', 'ocel:time', 'ocel_time', 'ocel_changed_field', 'ocel_id'}

def _atom_block_keys(oid2val: Dict[str, Any], type_oids: Set[str]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for oid in type_oids:
        if oid in oid2val:
            out[oid] = ('V', str(oid2val[oid]))
        else:
            out[oid] = ('M', oid)
    return out

def _refines(keys_a: Dict[str, Any], keys_b: Dict[str, Any], type_oids: Set[str]) -> bool:
    a_to_b: Dict[Any, Any] = {}
    for oid in type_oids:
        ka = keys_a[oid]
        kb = keys_b[oid]
        prev = a_to_b.get(ka, kb)
        if prev != kb:
            return False
        a_to_b[ka] = kb
    return True

def _compute_per_type_fd_dag(atoms: List[Tuple[str, Dict[str, Any]]], type_oids: Set[str]) -> Tuple[Dict[int, List[int]], Dict[int, List[int]]]:
    n = len(atoms)
    k_top = n + 1
    if n == 0:
        return ({0: [k_top], k_top: []}, {0: [], k_top: [0]})
    block_keys = [_atom_block_keys(m, type_oids) for _, m in atoms]
    refines = [[i == j for j in range(n)] for i in range(n)]
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if _refines(block_keys[i], block_keys[j], type_oids):
                refines[i][j] = True
    strict_lt = [[refines[i][j] and (not refines[j][i]) for j in range(n)] for i in range(n)]
    atom_cover_up: Dict[int, List[int]] = {i: [] for i in range(n)}
    atom_cover_dn: Dict[int, List[int]] = {i: [] for i in range(n)}
    for i in range(n):
        for j in range(n):
            if not strict_lt[i][j]:
                continue
            intermediate = any((strict_lt[i][m] and strict_lt[m][j] for m in range(n) if m != i and m != j))
            if not intermediate:
                atom_cover_up[i].append(j)
                atom_cover_dn[j].append(i)
    minimal = [i for i in range(n) if not atom_cover_dn[i]]
    maximal = [i for i in range(n) if not atom_cover_up[i]]
    cover_up: Dict[int, List[int]] = {}
    cover_dn: Dict[int, List[int]] = {}
    cover_up[0] = sorted((i + 1 for i in minimal))
    cover_dn[0] = []
    cover_up[k_top] = []
    cover_dn[k_top] = sorted((i + 1 for i in maximal))
    for i in range(n):
        L = i + 1
        ups = sorted((j + 1 for j in atom_cover_up[i]))
        dns = sorted((j + 1 for j in atom_cover_dn[i]))
        cover_up[L] = ups if ups else [k_top]
        cover_dn[L] = dns if dns else [0]
    return (cover_up, cover_dn)

def _print_functional_dependencies(per_type_attrs: Dict[str, List[Tuple[str, Dict[str, Any]]]], obj_types_map: Dict[str, str]) -> None:
    for tau, atoms in per_type_attrs.items():
        type_oids = {oid for oid, t in obj_types_map.items() if t == tau}
        if len(atoms) < 2:
            continue
        block_keys = [_atom_block_keys(m, type_oids) for _, m in atoms]
        print(f'[fd  ] {tau}')
        found = False
        for i, (a_name, _) in enumerate(atoms):
            for j, (b_name, _) in enumerate(atoms):
                if i == j:
                    continue
                if _refines(block_keys[i], block_keys[j], type_oids):
                    print(f'       {a_name}  →  {b_name}')
                    found = True
        if not found:
            print('no non-trivial FDs')

def _drop_equivalent_atoms(per_type_attrs: Dict[str, List[Tuple[str, Dict[str, Any]]]], obj_types_map: Dict[str, str], verbose: bool=False) -> Dict[str, List[Tuple[str, Dict[str, Any]]]]:
    cleaned: Dict[str, List[Tuple[str, Dict[str, Any]]]] = {}
    for tau, atoms in per_type_attrs.items():
        type_oids = {oid for oid, t in obj_types_map.items() if t == tau}
        if len(atoms) <= 1:
            cleaned[tau] = atoms
            continue
        block_keys = [_atom_block_keys(m, type_oids) for _, m in atoms]
        keep: List[int] = []
        dropped: Set[int] = set()
        for i, (a_name, _) in enumerate(atoms):
            if i in dropped:
                continue
            keep.append(i)
            for j, (b_name, _) in enumerate(atoms):
                if i == j or j in dropped:
                    continue
                a_to_b = _refines(block_keys[i], block_keys[j], type_oids)
                b_to_a = _refines(block_keys[j], block_keys[i], type_oids)
                if a_to_b and b_to_a:
                    dropped.add(j)
                    if verbose:
                        print(f'[fd] drop equivalent atom for {tau}: {b_name} ≡ {a_name}; keeping {a_name}')
        cleaned[tau] = [atoms[i] for i in keep]
    return cleaned

def _build_attribute_hierarchies(ocel_objects_df, obj_types_map: Dict[str, str], exclude_attrs: Sequence[str]=()) -> Tuple[Dict[str, ObjectTypeHierarchy], Dict[str, List[str]], Dict[str, List[Tuple[str, Dict[str, Any]]]]]:
    excl = set(exclude_attrs or ())
    df = ocel_objects_df
    all_attr_cols = [c for c in df.columns if c not in _STD_OBJ_COLS]
    per_type_attrs: Dict[str, List[Tuple[str, Dict[str, Any]]]] = defaultdict(list)
    for tau in sorted(set(obj_types_map.values())):
        type_oids = {oid for oid, t in obj_types_map.items() if t == tau}
        sub = df[df[OBJ_ID].isin(type_oids)]
        for c in all_attr_cols:
            if c in excl:
                continue
            if c not in sub.columns:
                continue
            col = sub[[OBJ_ID, c]].dropna()
            if col.empty:
                continue
            mapping: Dict[str, Any] = {}
            for oid, v in zip(col[OBJ_ID], col[c]):
                mapping[oid] = v
            if len({str(v) for v in mapping.values()}) < 2:
                continue
            per_type_attrs[tau].append((c, mapping))
    per_type_attrs = defaultdict(list, _drop_equivalent_atoms(dict(per_type_attrs), obj_types_map, verbose=True))
    hierarchies: Dict[str, ObjectTypeHierarchy] = {}
    level_names: Dict[str, List[str]] = {}
    all_types = sorted(set(obj_types_map.values()))
    for tau in all_types:
        type_oids = {oid for oid, t in obj_types_map.items() if t == tau}
        atoms = per_type_attrs.get(tau, [])
        levels: List[Set[Any]] = [type_oids]
        mappings: List[Callable] = []
        names: List[str] = ['id']
        for attr, oid2val in atoms:
            mappings.append(lambda o, m=oid2val: m.get(o) if o in m else f'?{attr}={o!s}')
            val_set = {str(v) for v in oid2val.values()}
            levels.append(val_set)
            names.append(attr)
        mappings.append(lambda _o, t=tau: t)
        levels.append({tau})
        names.append('type')
        cover_up, cover_dn = _compute_per_type_fd_dag(atoms, type_oids)
        hierarchies[tau] = ObjectTypeHierarchy(object_type=tau, levels=levels, mappings=mappings, cover_up=cover_up, cover_dn=cover_dn)
        level_names[tau] = names
    return (hierarchies, level_names, dict(per_type_attrs))

def rebuild_hierarchies_from_attrs(per_type_attrs: Dict[str, List[Tuple[str, Dict[str, Any]]]], obj_types_map: Dict[str, str]) -> Tuple[Dict[str, ObjectTypeHierarchy], Dict[str, List[str]]]:
    hierarchies: Dict[str, ObjectTypeHierarchy] = {}
    level_names: Dict[str, List[str]] = {}
    all_types = sorted(set(obj_types_map.values()))
    for tau in all_types:
        type_oids = {oid for oid, t in obj_types_map.items() if t == tau}
        atoms = per_type_attrs.get(tau, [])
        levels: List[Set[Any]] = [type_oids]
        mappings: List[Callable] = []
        names: List[str] = ['id']
        for attr, oid2val in atoms:
            mappings.append(lambda o, m=oid2val, a=attr: m.get(o) if o in m else f'?{a}={o!s}')
            levels.append({str(v) for v in oid2val.values()})
            names.append(attr)
        mappings.append(lambda _o, t=tau: t)
        levels.append({tau})
        names.append('type')
        cover_up, cover_dn = _compute_per_type_fd_dag(atoms, type_oids)
        hierarchies[tau] = ObjectTypeHierarchy(object_type=tau, levels=levels, mappings=mappings, cover_up=cover_up, cover_dn=cover_dn)
        level_names[tau] = names
    return (hierarchies, level_names)

@dataclass
class CfgRun:
    cfg: Tuple[int, ...]
    behaviors: List[BehaviorGraph]
    iso_ids: List[int]
    K: int
    n_executions: int
    behavior_time_s: float = 0.0
    iso_time_s: float = 0.0
    is_interesting: bool = False
    reason_skipped: str = ''

@dataclass
class CfgTimingRow:
    dataset: str
    cfg: str
    phi: str
    interesting: bool
    reason_skipped: str
    K: int
    n_executions: int
    K_ratio: float
    cfg_height: int
    cfg_height_norm: float
    avg_behavior_size: float
    support_min: int
    support_mean: float
    support_max: int
    reduction_from_bottom: float
    compression_from_bottom: float
    behavior_time_s: float
    mining_time_s: Optional[float] = None
    miner: str = ''
    n_graphs: int = 0
    avg_mining_graph_size: float = 0.0
    total_mining_graph_size: int = 0
    min_support_effective: int = 0
    n_patterns_raw: int = 0
    n_patterns_in_window: int = 0
    pattern_support_min: int = 0
    pattern_support_mean: float = 0.0
    pattern_support_median: float = 0.0
    pattern_support_max: int = 0
    pattern_support_ratio_mean: float = 0.0
    pattern_support_ratio_median: float = 0.0
    pattern_support_ratio_max: float = 0.0
    high_support_threshold_ratio: float = 0.25
    high_support_threshold_abs: int = 0
    n_high_support_patterns: int = 0
    high_support_pattern_ratio: float = 0.0

def _top_cfg(types, hierarchies) -> Tuple[int, ...]:
    return tuple((hierarchies[tau].k for tau in types))

def _lower_covers(cfg: Tuple[int, ...], types: List[str], hierarchies: Dict[str, ObjectTypeHierarchy]) -> List[Tuple[int, ...]]:
    out = []
    for i, tau in enumerate(types):
        h = hierarchies[tau]
        for lo in h.cover_dn.get(cfg[i], []):
            c = list(cfg)
            c[i] = lo
            out.append(tuple(c))
    return out

def _enumerate_configs_top_down(types: List[str], hierarchies: Dict[str, ObjectTypeHierarchy]) -> List[Tuple[int, ...]]:
    top = _top_cfg(types, hierarchies)
    seen: Set[Tuple[int, ...]] = {top}
    order: List[Tuple[int, ...]] = []
    queue: List[Tuple[int, ...]] = [top]
    while queue:
        cfg = queue.pop(0)
        order.append(cfg)
        for child in sorted(_lower_covers(cfg, types, hierarchies)):
            if child not in seen:
                seen.add(child)
                queue.append(child)
    return order

def _compute_cfg(cfg: Tuple[int, ...], types: List[str], executions: List[ProcessExecution], hierarchies: Dict[str, ObjectTypeHierarchy], obj_types_map: Dict[str, str]) -> CfgRun:
    lc = LevelConfiguration(dict(zip(types, cfg)), hierarchies, obj_types_map)
    t_beh = time.perf_counter()
    behs = [compute_behavior(OCELLevelGraph(ex, lc)) for ex in executions]
    behavior_time_s = time.perf_counter() - t_beh
    t_iso = time.time()
    ids = compute_iso_classes(behs)
    iso_time_s = time.time() - t_iso
    K = len(set(ids))
    return CfgRun(cfg=cfg, behaviors=behs, iso_ids=ids, K=K, n_executions=len(executions), behavior_time_s=behavior_time_s, iso_time_s=iso_time_s)

def _is_interesting(run: CfgRun) -> Tuple[bool, str]:
    if run.n_executions == 0:
        return (False, 'empty log')
    if run.K == run.n_executions:
        return (False, 'no merging (K = n_executions)')
    if run.K <= 1:
        return (False, 'single trivial class')
    return (True, '')

def _pattern_support_set(pattern: nx.DiGraph, graphs: Sequence[nx.DiGraph]) -> Set[int]:
    hits: Set[int] = set()
    for i, G in enumerate(graphs):
        if _matcher(G, pattern).subgraph_is_isomorphic():
            hits.add(i)
    return hits

@dataclass
class BundlePattern:
    cfg: Tuple[int, ...]
    pattern: nx.DiGraph
    signature: Tuple
    support: int
    n_graphs: int
    size_nodes: int
    size_edges: int
    in_exec_idx: List[int]
    out_exec_idx: List[int]
    kpi_stats: Dict[str, Dict[str, float]]

@dataclass
class BundleCfgRun:
    cfg: Tuple[int, ...]
    K: int
    n_executions: int
    is_interesting: bool
    reason_skipped: str
    iso_ids: List[int]
    reduction_from_bottom: float = 0.0
    compression_from_bottom: float = 0.0
    avg_behavior_size: float = 0.0
    patterns: List[BundlePattern] = field(default_factory=list)

@dataclass
class RunBundle:
    ocel_path: str
    leading_type: str
    timestamp_iso: str
    elapsed_s: float
    args: Dict[str, Any]
    types: List[str]
    obj_types_map: Dict[str, str]
    level_names: Dict[str, List[str]]
    per_type_attrs: Dict[str, List[Tuple[str, Dict[str, Any]]]]
    executions: List[ProcessExecution]
    kpi_specs: List[Dict[str, str]]
    kpi_values: Dict[str, List[float]]
    primary_kpi: str
    cfg_runs: List[BundleCfgRun]
    n_events: int
    n_objects: int
    n_cfg_total: int
    n_cfg_interesting: int
    n_cfg_skipped: int
    n_patterns_total: int

def save_bundle(bundle: RunBundle, path: str) -> None:
    with open(path, 'wb') as f:
        pickle.dump(bundle, f, protocol=pickle.HIGHEST_PROTOCOL)

def load_bundle(path: str) -> RunBundle:
    with open(path, 'rb') as f:
        return pickle.load(f)

@dataclass
class LiftedPattern:
    cfg: Tuple[int, ...]
    pattern: nx.DiGraph
    signature: Tuple
    support: int
    n_graphs: int
    size_nodes: int
    size_edges: int
    in_exec_idx: List[int]
    out_exec_idx: List[int]
    mu_in: float
    mu_out: float
    delta: float
    signed_delta: float

def _structural_canonical_form(P: nx.DiGraph) -> Tuple[Tuple, Tuple[Tuple[int, ...], List[int]]]:
    nodes = list(P.nodes())
    node_labels = [P.nodes[n]['label'] for n in nodes]
    edges_raw = [(nodes.index(u), nodes.index(v)) for u, v in P.edges()]
    n = len(nodes)
    best: Optional[Tuple] = None
    best_perm: Optional[Tuple[Tuple[int, ...], List[int]]] = None
    for perm in itertools.permutations(range(n)):
        inv = [0] * n
        for new_i, old_i in enumerate(perm):
            inv[old_i] = new_i
        nl = tuple((node_labels[perm[i]] for i in range(n)))
        el = tuple(sorted(((inv[u], inv[v]) for u, v in edges_raw)))
        sig = (nl, el)
        if best is None or sig < best:
            best = sig
            best_perm = (perm, inv)
    assert best is not None and best_perm is not None
    return (best, best_perm)

def _merge_equivalent_bundle_patterns(bundled: List[BundlePattern], verbose: bool=False) -> Tuple[List[BundlePattern], int]:
    if not bundled:
        return (bundled, 0)
    groups: Dict[Tuple, List[Tuple[int, BundlePattern, Tuple, Tuple[Tuple[int, ...], List[int]]]]] = defaultdict(list)
    for i, bp in enumerate(bundled):
        try:
            sig, perm = _structural_canonical_form(bp.pattern)
        except Exception:
            sig = ('__opaque__', i)
            perm = (tuple(range(bp.pattern.number_of_nodes())), list(range(bp.pattern.number_of_nodes())))
        gkey = (sig, tuple(bp.in_exec_idx))
        groups[gkey].append((i, bp, sig, perm))
    merged_list: List[BundlePattern] = []
    n_collapsed = 0
    for gkey, items in groups.items():
        if len(items) == 1:
            merged_list.append(items[0][1])
            continue
        canon_node_labels = items[0][2][0]
        canon_edges_raw = items[0][2][1]
        edge_labels: Dict[Tuple[int, int], Set[str]] = defaultdict(set)
        for _, bp, _sig, (_perm, inv) in items:
            nodes = list(bp.pattern.nodes())
            for u, v, d in bp.pattern.edges(data=True):
                cu = inv[nodes.index(u)]
                cv = inv[nodes.index(v)]
                lbls = d.get('labels')
                if lbls is None:
                    lbl = d.get('label')
                    if lbl is not None:
                        edge_labels[cu, cv].add(str(lbl))
                else:
                    edge_labels[cu, cv].update((str(x) for x in lbls))
        Pmerged: nx.DiGraph = nx.DiGraph()
        for i, lbl in enumerate(canon_node_labels):
            Pmerged.add_node(i, label=lbl)
        for (cu, cv), lbls in edge_labels.items():
            Pmerged.add_edge(cu, cv, labels=frozenset(lbls))
        new_sig = canonical_signature(Pmerged)
        size_edges = sum((len(d['labels']) for _, _, d in Pmerged.edges(data=True)))
        bp0 = items[0][1]
        merged_list.append(BundlePattern(cfg=bp0.cfg, pattern=Pmerged, signature=new_sig, support=bp0.support, n_graphs=bp0.n_graphs, size_nodes=Pmerged.number_of_nodes(), size_edges=size_edges, in_exec_idx=bp0.in_exec_idx, out_exec_idx=bp0.out_exec_idx, kpi_stats=bp0.kpi_stats))
        n_collapsed += len(items) - 1
    if verbose and n_collapsed > 0:
        print(f'[merge] collapsed {n_collapsed} duplicate-edge-label patterns into {len(merged_list)} entries (was {len(bundled)})')
    return (merged_list, n_collapsed)

def _merge_equivalent_lifted_patterns(lifted: List[LiftedPattern], verbose: bool=False) -> List[LiftedPattern]:
    if not lifted:
        return lifted
    groups: Dict[Tuple, List[Tuple[int, LiftedPattern, Tuple, Tuple[Tuple[int, ...], List[int]]]]] = defaultdict(list)
    for i, lp in enumerate(lifted):
        try:
            sig, perm = _structural_canonical_form(lp.pattern)
        except Exception:
            sig = ('__opaque__', i)
            perm = (tuple(range(lp.pattern.number_of_nodes())), list(range(lp.pattern.number_of_nodes())))
        gkey = (sig, tuple(lp.in_exec_idx))
        groups[gkey].append((i, lp, sig, perm))
    out: List[LiftedPattern] = []
    for items in groups.values():
        if len(items) == 1:
            out.append(items[0][1])
            continue
        canon_node_labels = items[0][2][0]
        edge_labels: Dict[Tuple[int, int], Set[str]] = defaultdict(set)
        for _, lp, _sig, (_perm, inv) in items:
            nodes = list(lp.pattern.nodes())
            for u, v, d in lp.pattern.edges(data=True):
                cu = inv[nodes.index(u)]
                cv = inv[nodes.index(v)]
                lbls = d.get('labels')
                if lbls is None:
                    lbl = d.get('label')
                    if lbl is not None:
                        edge_labels[cu, cv].add(str(lbl))
                else:
                    edge_labels[cu, cv].update((str(x) for x in lbls))
        Pmerged: nx.DiGraph = nx.DiGraph()
        for i, lbl in enumerate(canon_node_labels):
            Pmerged.add_node(i, label=lbl)
        for (cu, cv), lbls in edge_labels.items():
            Pmerged.add_edge(cu, cv, labels=frozenset(lbls))
        size_edges = sum((len(d['labels']) for _, _, d in Pmerged.edges(data=True)))
        new_sig = canonical_signature(Pmerged)
        lp0 = items[0][1]
        out.append(LiftedPattern(cfg=lp0.cfg, pattern=Pmerged, signature=new_sig, support=lp0.support, n_graphs=lp0.n_graphs, size_nodes=Pmerged.number_of_nodes(), size_edges=size_edges, in_exec_idx=lp0.in_exec_idx, out_exec_idx=lp0.out_exec_idx, mu_in=lp0.mu_in, mu_out=lp0.mu_out, delta=lp0.delta, signed_delta=lp0.signed_delta))
    if verbose and len(out) < len(lifted):
        print(f'[merge] lifted: {len(lifted)} → {len(out)} after multi-label-edge merge')
    return out

def _mine_and_score(run: CfgRun, kpi_values: Dict[str, List[float]], primary_kpi: str, s_min_abs: int, s_max_abs: int, max_edges: int, beam_width: int, min_support: int, top_k_per_cfg: int, verbose: bool, collect_all: bool=False, miner: str='gspan') -> Tuple[List[LiftedPattern], List[BundlePattern], Dict[str, Any]]:
    graphs: List[nx.DiGraph] = []
    g_to_ex: List[int] = []
    for i, beh in enumerate(run.behaviors):
        if beh.abstract_nodes:
            graphs.append(behavior_to_nx(beh))
            g_to_ex.append(i)
    n_g = len(graphs)
    if n_g == 0:
        return ([], [], {'mining_time_s': 0.0, 'miner': miner, 'n_graphs': 0, 'avg_mining_graph_size': 0.0, 'total_mining_graph_size': 0, 'min_support_effective': max(min_support, s_min_abs), 'n_patterns_raw': 0, 'n_patterns_in_window': 0})
    effective_min = max(min_support, s_min_abs)
    UNBOUNDED_K = 10 ** 9
    t_mining = time.perf_counter()
    all_patterns = mine_dispatch(miner, graphs, k=UNBOUNDED_K, beam_width=beam_width, max_edges=max_edges, min_support=effective_min, verbose=verbose)
    mining_time_s = time.perf_counter() - t_mining
    for mp in all_patterns[:20]:
        print('[debug all]', 'support=', mp.support, 'nodes=', mp.size_nodes, 'edges=', mp.size_edges, 'signature=', mp.signature)
    in_window = [mp for mp in all_patterns if s_min_abs <= mp.support <= s_max_abs]
    supports = [mp.support for mp in in_window]
    if supports:
        pattern_support_min = min(supports)
        pattern_support_max = max(supports)
        pattern_support_mean = sum(supports) / len(supports)
        supports_sorted = sorted(supports)
        mid = len(supports_sorted) // 2
        if len(supports_sorted) % 2 == 1:
            pattern_support_median = supports_sorted[mid]
        else:
            pattern_support_median = (supports_sorted[mid - 1] + supports_sorted[mid]) / 2
    else:
        pattern_support_min = 0
        pattern_support_max = 0
        pattern_support_mean = 0.0
        pattern_support_median = 0.0
    n_exec = run.n_executions if run.n_executions else 1
    pattern_support_ratio_mean = pattern_support_mean / n_exec
    pattern_support_ratio_max = pattern_support_max / n_exec
    pattern_support_ratio_median = pattern_support_median / n_exec
    HIGH_SUPPORT_RATIO = 0.25
    high_support_threshold_abs = max(1, int(HIGH_SUPPORT_RATIO * n_exec))
    n_high_support_patterns = sum((1 for s in supports if s >= high_support_threshold_abs))
    high_support_pattern_ratio = n_high_support_patterns / len(supports) if supports else 0.0
    total_mining_graph_size = sum((g.number_of_nodes() + g.number_of_edges() for g in graphs))
    avg_mining_graph_size = total_mining_graph_size / n_g if n_g else 0.0
    mining_stats = {'mining_time_s': mining_time_s, 'miner': miner, 'n_graphs': n_g, 'avg_mining_graph_size': avg_mining_graph_size, 'total_mining_graph_size': total_mining_graph_size, 'min_support_effective': effective_min, 'n_patterns_raw': len(all_patterns), 'n_patterns_in_window': len(in_window), 'pattern_support_min': pattern_support_min, 'pattern_support_mean': pattern_support_mean, 'pattern_support_median': pattern_support_median, 'pattern_support_max': pattern_support_max, 'pattern_support_ratio_mean': pattern_support_ratio_mean, 'pattern_support_ratio_median': pattern_support_ratio_median, 'pattern_support_ratio_max': pattern_support_ratio_max, 'high_support_threshold_ratio': HIGH_SUPPORT_RATIO, 'high_support_threshold_abs': high_support_threshold_abs, 'n_high_support_patterns': n_high_support_patterns, 'high_support_pattern_ratio': high_support_pattern_ratio}
    if verbose:
        sups = sorted((mp.support for mp in all_patterns), reverse=True)
        if sups:
            q = lambda p: sups[min(len(sups) - 1, int(round((len(sups) - 1) * p)))]
            sup_dist = f'min={sups[-1]}  q25={q(0.75)}  median={q(0.5)}  q75={q(0.25)}  max={sups[0]}'
        else:
            sup_dist = '(empty)'
        print(f'mined={len(all_patterns)} in_window={len(in_window)} (window=[{s_min_abs},{s_max_abs}], n_g={n_g})')
        print(f'support distribution: {sup_dist}')
    primary = kpi_values[primary_kpi]
    scored: List[LiftedPattern] = []
    bundled: List[BundlePattern] = []
    for mp in in_window:
        hit_g = _pattern_support_set(mp.pattern, graphs)
        print('[debug pattern]', 'mp.support=', mp.support, 'nx_support=', len(hit_g), 'nodes=', mp.size_nodes, 'edges=', mp.size_edges)
        in_exec = sorted((g_to_ex[i] for i in hit_g))
        all_ex = list(range(run.n_executions))
        in_set = set(in_exec)
        out_exec = [i for i in all_ex if i not in in_set]
        if not in_exec or not out_exec:
            continue
        mu_in = mean((primary[i] for i in in_exec))
        mu_out = mean((primary[i] for i in out_exec))
        signed = mu_in - mu_out
        delta = abs(signed)
        scored.append(LiftedPattern(cfg=run.cfg, pattern=mp.pattern, signature=mp.signature, support=mp.support, n_graphs=n_g, size_nodes=mp.size_nodes, size_edges=mp.size_edges, in_exec_idx=in_exec, out_exec_idx=out_exec, mu_in=mu_in, mu_out=mu_out, delta=delta, signed_delta=signed))
        if collect_all:
            kpi_stats: Dict[str, Dict[str, float]] = {}
            for kname, kvals in kpi_values.items():
                mi = mean((kvals[i] for i in in_exec))
                mo = mean((kvals[i] for i in out_exec))
                kpi_stats[kname] = {'mu_in': mi, 'mu_out': mo, 'signed_delta': mi - mo, 'delta': abs(mi - mo)}
            bundled.append(BundlePattern(cfg=run.cfg, pattern=mp.pattern, signature=mp.signature, support=mp.support, n_graphs=n_g, size_nodes=mp.size_nodes, size_edges=mp.size_edges, in_exec_idx=in_exec, out_exec_idx=out_exec, kpi_stats=kpi_stats))
    scored = _merge_equivalent_lifted_patterns(scored, verbose=verbose)
    if collect_all:
        bundled, _n_collapsed = _merge_equivalent_bundle_patterns(bundled, verbose=verbose)
    bundled.sort(key=lambda b: -b.kpi_stats.get(primary_kpi, {}).get('delta', 0.0))
    scored.sort(key=lambda p: -p.delta)
    if top_k_per_cfg > 0:
        scored = scored[:top_k_per_cfg]
    return (scored, bundled, mining_stats)
_HTML_TEMPLATE = '<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n<title>KPI — top-{{ k }} KPI-discriminating patterns</title>\n<script src="https://unpkg.com/cytoscape@3.28.1/dist/cytoscape.min.js"></script>\n<script src="https://unpkg.com/dagre@0.8.5/dist/dagre.min.js"></script>\n<script src="https://unpkg.com/cytoscape-dagre@2.5.0/cytoscape-dagre.js"></script>\n<style>\n  :root {\n    --bg:#f7f7fb; --card:#fff; --ink:#1f2937;\n    --muted:#6b7280; --border:#e5e7eb;\n    --pos:#C13B3B; --neg:#2E7D32;\n  }\n  * { box-sizing:border-box; }\n  body { margin:0; font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;\n         background:var(--bg); color:var(--ink); }\n  header { padding:20px 28px; background:#fff; border-bottom:1px solid var(--border);\n           position:sticky; top:0; z-index:5; }\n  header h1 { margin:0 0 6px 0; font-size:18px; font-weight:600; }\n  header .meta { color:var(--muted); font-size:13px; }\n  header code { background:#f1f1f5; padding:1px 6px; border-radius:4px; }\n\n  main { padding:14px 28px 32px; display:flex; flex-direction:column; gap:18px; }\n  .pattern-card {\n    background:var(--card); border:1px solid var(--border);\n    border-radius:10px; padding:14px 18px; box-shadow:0 1px 3px rgba(0,0,0,.04);\n  }\n  .pattern-card h3 {\n    margin:0 0 6px 0; font-size:15px; font-weight:600;\n    display:flex; justify-content:space-between; align-items:baseline;\n  }\n  .rank { color:var(--muted); font-weight:500; font-size:12px; }\n  .phi  {\n    font-family:monospace; font-size:12px;\n    background:#f9fafb; border:1px solid var(--border); border-radius:6px;\n    padding:6px 8px; margin-bottom:8px; word-break:break-word;\n  }\n  .metrics {\n    display:flex; gap:18px; color:var(--muted); font-size:13px; margin-bottom:10px;\n    flex-wrap:wrap;\n  }\n  .metrics b { color:var(--ink); font-weight:600; }\n  .delta-up   { color:var(--pos); font-weight:700; }\n  .delta-down { color:var(--neg); font-weight:700; }\n  .cy { width:100%; height:320px; background:#fafbff;\n        border:1px dashed var(--border); border-radius:6px; }\n  .empty { color:var(--muted); font-style:italic; padding:24px; text-align:center;\n           border:1px dashed var(--border); border-radius:8px; background:var(--card); }\n</style>\n</head>\n<body>\n<header>\n  <h1>KPI — KPI-discriminating subgraph patterns</h1>\n  <div class="meta">\n    OCEL: <code>{{ ocel_path }}</code>\n    &nbsp;·&nbsp; leading type = <b>{{ leading_type }}</b>\n    &nbsp;·&nbsp; executions = <b>{{ n_executions }}</b>\n    &nbsp;·&nbsp; Φ evaluated = <b>{{ n_cfg_eval }}</b>\n    &nbsp;·&nbsp; Φ interesting = <b>{{ n_cfg_interesting }}</b>\n    &nbsp;·&nbsp; support window = <b>[{{ s_min }}, {{ s_max }}]</b>\n    &nbsp;·&nbsp; KPI = <b>{{ kpi_label }}</b>\n    &nbsp;·&nbsp; elapsed = <b>{{ elapsed_s }} s</b>\n  </div>\n</header>\n\n<main>\n  {% if patterns %}\n  {% for p in patterns %}\n  <div class="pattern-card">\n    <h3>\n      <span>Pattern {{ loop.index }}\n        <span class="rank"> — support {{ p.support }}/{{ p.n_graphs }}\n              ({{ (100*p.support/p.n_graphs)|round(1) }}%)</span>\n      </span>\n      <span class="{{ \'delta-up\' if p.signed_delta >= 0 else \'delta-down\' }}">\n        KPI = {{ p.delta_h }}\n        ({{ \'+\' if p.signed_delta >= 0 else \'−\' }}{{ p.signed_delta_h_abs }})\n      </span>\n    </h3>\n    <div class="phi">Φ = {{ p.phi_str }}</div>\n    <div class="metrics">\n      <span>μ<sub>in</sub> = <b>{{ p.mu_in_h }}</b></span>\n      <span>μ<sub>out</sub> = <b>{{ p.mu_out_h }}</b></span>\n      <span>|E<sub>in</sub>| = <b>{{ p.in_exec_idx|length }}</b></span>\n      <span>|E<sub>out</sub>| = <b>{{ p.out_exec_idx|length }}</b></span>\n      <span><b>{{ p.size_nodes }}</b> nodes</span>\n      <span><b>{{ p.size_edges }}</b> edges</span>\n    </div>\n    <div class="cy" id="cy-{{ loop.index0 }}"></div>\n  </div>\n  {% endfor %}\n  {% else %}\n  <div class="empty">No interesting patterns found with the current parameters.</div>\n  {% endif %}\n</main>\n\n<script>\n  const PATTERNS = {{ patterns_json | safe }};\n  const baseStyle = [\n    { selector:"node", style:{\n      "background-color":"#4E79A7","label":"data(label)","color":"#fff",\n      "font-size":"11px","font-family":"monospace","text-wrap":"wrap",\n      "text-max-width":"170px","text-valign":"center","text-halign":"center",\n      "width":"170px","height":"58px","shape":"roundrectangle",\n    }},\n    { selector:"edge", style:{\n      "curve-style":"bezier","control-point-step-size":40,\n      "target-arrow-shape":"triangle","target-arrow-color":"#6b7280",\n      "line-color":"#6b7280","width":2,"label":"data(label)","font-size":"10px",\n      "font-family":"monospace","color":"#374151",\n      "text-background-color":"#fff","text-background-opacity":0.9,\n      "text-background-padding":"3px","text-rotation":"autorotate",\n    }},\n  ];\n  PATTERNS.forEach((elems, i) => {\n    const ss = baseStyle.map(s => ({...s, style: {...s.style}}));\n    ss[0].style["background-color"] = elems.accent;\n    const el = document.getElementById(`cy-${i}`);\n    if (!el) return;\n    cytoscape({\n      container: el, elements: elems.elements, style: ss,\n      layout:{ name:"dagre", rankDir:"LR", nodeSep:28, rankSep:60 },\n      userZoomingEnabled:true, userPanningEnabled:true,\n      boxSelectionEnabled:false,\n    });\n  });\n</script>\n</body>\n</html>\n'

def _fmt_cfg(cfg: Tuple[int, ...], types: List[str], level_names: Dict[str, List[str]]) -> str:
    parts = []
    for i, tau in enumerate(types):
        names = level_names.get(tau, [])
        lv = cfg[i]
        lbl = names[lv] if 0 <= lv < len(names) else str(lv)
        parts.append(f'{tau}: cfg = {lbl}')
    return '  │  '.join(parts)

def _fmt_duration_seconds(s: float) -> str:
    abs_s = abs(s)
    if abs_s >= 86400:
        return f'{s / 86400:.2f} d'
    if abs_s >= 3600:
        return f'{s / 3600:.2f} h'
    if abs_s >= 60:
        return f'{s / 60:.1f} min'
    return f'{s:.1f} s'

def _pattern_to_cy_elements(P: nx.DiGraph) -> List[Dict]:
    nodes = [{'data': {'id': f'n{n}', 'label': d['label']}} for n, d in P.nodes(data=True)]
    edges: List[Dict] = []
    eid = 0
    for u, v, d in P.edges(data=True):
        for lbl in sorted(d['labels']):
            edges.append({'data': {'id': f'e{eid}', 'source': f'n{u}', 'target': f'n{v}', 'label': lbl}})
            eid += 1
    return nodes + edges

def render_html(*, ocel_path: str, leading_type: str, n_executions: int, n_cfg_eval: int, n_cfg_interesting: int, s_min: float, s_max: float, kpi_label: str, elapsed_s: float, patterns: List[Dict[str, Any]], out_path: str, k: int) -> None:
    from jinja2 import Template
    patterns_js = [{'accent': p['accent'], 'elements': _pattern_to_cy_elements(p['pattern'])} for p in patterns]
    html_text = Template(_HTML_TEMPLATE).render(ocel_path=html.escape(ocel_path), leading_type=leading_type, n_executions=n_executions, n_cfg_eval=n_cfg_eval, n_cfg_interesting=n_cfg_interesting, s_min=s_min, s_max=s_max, kpi_label=kpi_label, elapsed_s=f'{elapsed_s:.1f}', patterns=patterns, patterns_json=json.dumps(patterns_js), k=k)
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(html_text)

@dataclass
class RunStats:
    n_events: int
    n_objects: int
    n_executions: int
    types: List[str]
    n_cfg_total: int
    n_cfg_interesting: int
    n_cfg_skipped: int
    n_patterns_total: int
    n_patterns_feas: int

@dataclass(frozen=True)
class LatticeNodeMetrics:
    K: int
    s: float
    sup_min: int
    sup_max: int
    sup_mean: float

def _behavior_size(b: BehaviorGraph) -> int:
    return len(b.abstract_nodes) + len(b.abstract_edges)

def _cfg_height(cfg: Tuple[int, ...], types: List[str], hierarchies: Dict[str, ObjectTypeHierarchy]) -> int:
    return sum((hierarchies[types[i]].height(cfg[i]) for i in range(len(types))))

def save_cfg_timing_csv(path: str, rows: List[CfgTimingRow]) -> None:
    fieldnames = ['dataset', 'cfg', 'phi', 'interesting', 'reason_skipped', 'K', 'n_executions', 'K_ratio', 'cfg_height', 'cfg_height_norm', 'avg_behavior_size', 'support_min', 'support_mean', 'support_max', 'reduction_from_bottom', 'compression_from_bottom', 'behavior_time_s', 'mining_time_s', 'miner', 'n_graphs', 'avg_mining_graph_size', 'total_mining_graph_size', 'min_support_effective', 'n_patterns_raw', 'n_patterns_in_window', 'pattern_support_min', 'pattern_support_mean', 'pattern_support_median', 'pattern_support_max', 'pattern_support_ratio_mean', 'pattern_support_ratio_median', 'pattern_support_ratio_max', 'high_support_threshold_ratio', 'high_support_threshold_abs', 'n_high_support_patterns', 'high_support_pattern_ratio']
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({'dataset': r.dataset, 'cfg': r.cfg, 'phi': r.phi, 'interesting': r.interesting, 'reason_skipped': r.reason_skipped, 'K': r.K, 'n_executions': r.n_executions, 'K_ratio': r.K_ratio, 'cfg_height': r.cfg_height, 'cfg_height_norm': r.cfg_height_norm, 'avg_behavior_size': r.avg_behavior_size, 'support_min': r.support_min, 'support_mean': r.support_mean, 'support_max': r.support_max, 'reduction_from_bottom': r.reduction_from_bottom, 'compression_from_bottom': r.compression_from_bottom, 'behavior_time_s': r.behavior_time_s, 'mining_time_s': '' if r.mining_time_s is None else r.mining_time_s, 'miner': r.miner, 'n_graphs': r.n_graphs, 'avg_mining_graph_size': r.avg_mining_graph_size, 'total_mining_graph_size': r.total_mining_graph_size, 'min_support_effective': r.min_support_effective, 'n_patterns_raw': r.n_patterns_raw, 'n_patterns_in_window': r.n_patterns_in_window, 'pattern_support_min': r.pattern_support_min, 'pattern_support_mean': r.pattern_support_mean, 'pattern_support_median': r.pattern_support_median, 'pattern_support_max': r.pattern_support_max, 'pattern_support_ratio_mean': r.pattern_support_ratio_mean, 'pattern_support_ratio_median': r.pattern_support_ratio_median, 'pattern_support_ratio_max': r.pattern_support_ratio_max, 'high_support_threshold_ratio': r.high_support_threshold_ratio, 'high_support_threshold_abs': r.high_support_threshold_abs, 'n_high_support_patterns': r.n_high_support_patterns, 'high_support_pattern_ratio': r.high_support_pattern_ratio})

def behavior_reduction(metrics: Dict[Tuple, LatticeNodeMetrics], u: Tuple, v: Tuple) -> float:
    Ku = metrics[u].K
    if Ku == 0:
        return float('nan')
    return 1.0 - metrics[v].K / Ku

def behavior_compression(metrics: Dict[Tuple, LatticeNodeMetrics], u: Tuple, v: Tuple) -> float:
    su = metrics[u].s
    if su == 0:
        return float('nan')
    return 1.0 - metrics[v].s / su

def save_cfg_metrics_csv(path: str, *, runs: List[CfgRun], types: List[str], level_names: Dict[str, List[str]], lattice_metrics: Dict[Tuple, LatticeNodeMetrics], bottom_cfg: Tuple[int, ...]) -> None:
    rows = []
    for run in runs:
        m = lattice_metrics[run.cfg]
        red = behavior_reduction(lattice_metrics, bottom_cfg, run.cfg)
        comp = behavior_compression(lattice_metrics, bottom_cfg, run.cfg)
        rows.append({'cfg': repr(run.cfg), 'phi': _fmt_cfg(run.cfg, types, level_names), 'interesting': run.is_interesting, 'reason_skipped': run.reason_skipped, 'K': run.K, 'n_executions': run.n_executions, 'avg_behavior_size': m.s, 'support_min': m.sup_min, 'support_mean': m.sup_mean, 'support_max': m.sup_max, 'reduction_from_bottom': red, 'compression_from_bottom': comp})
    with open(path, 'w', newline='', encoding='utf-8') as f:
        fieldnames = list(rows[0].keys()) if rows else ['cfg', 'phi', 'interesting', 'reason_skipped', 'K', 'n_executions', 'avg_behavior_size', 'support_min', 'support_mean', 'support_max', 'reduction_from_bottom', 'compression_from_bottom']
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

def _pct(x: float) -> str:
    return 'nan' if x != x else f'{x * 100:+.1f}%'

def save_run_stats_csv(path: str, *, timings: Dict[str, float], elapsed: float, types: List[str], level_names: Dict[str, List[str]], lattice_metrics: Dict[Tuple, LatticeNodeMetrics], bottom_cfg: Tuple[int, ...], top_cfg: Tuple[int, ...], n_events: int, n_objects: int, n_executions: int, n_cfg_total: int, n_cfg_interesting: int, n_cfg_mineable: int, n_cfg_skipped: int, n_patterns_total: int, n_patterns_top: int) -> None:
    rows = []
    for name, sec in timings.items():
        rows.append({'section': 'timing', 'name': name, 'value': sec, 'unit': 'seconds', 'cfg': '', 'K': '', 'avg_behavior_size': '', 'support_min': '', 'support_mean': '', 'support_max': '', 'reduction_from_bottom': '', 'compression_from_bottom': ''})
    rows.append({'section': 'timing', 'name': 'total', 'value': elapsed, 'unit': 'seconds', 'cfg': '', 'K': '', 'avg_behavior_size': '', 'support_min': '', 'support_mean': '', 'support_max': '', 'reduction_from_bottom': '', 'compression_from_bottom': ''})
    summary_values = {'n_events': n_events, 'n_objects': n_objects, 'n_executions': n_executions, 'n_object_types': len(types), 'n_cfg_total': n_cfg_total, 'n_cfg_interesting': n_cfg_interesting, 'n_cfg_mineable': n_cfg_mineable, 'n_cfg_skipped': n_cfg_skipped, 'n_patterns_total': n_patterns_total, 'n_patterns_top': n_patterns_top}
    for name, value in summary_values.items():
        rows.append({'section': 'summary', 'name': name, 'value': value, 'unit': 'count', 'cfg': '', 'K': '', 'avg_behavior_size': '', 'support_min': '', 'support_mean': '', 'support_max': '', 'reduction_from_bottom': '', 'compression_from_bottom': ''})
    for label, cfg in [('bottom', bottom_cfg), ('top', top_cfg)]:
        m = lattice_metrics[cfg]
        red = behavior_reduction(lattice_metrics, bottom_cfg, cfg)
        comp = behavior_compression(lattice_metrics, bottom_cfg, cfg)
        rows.append({'section': 'lattice_node', 'name': label, 'value': '', 'unit': '', 'cfg': _fmt_cfg(cfg, types, level_names), 'K': m.K, 'avg_behavior_size': m.s, 'support_min': m.sup_min, 'support_mean': m.sup_mean, 'support_max': m.sup_max, 'reduction_from_bottom': red, 'compression_from_bottom': comp})
    fieldnames = ['section', 'name', 'value', 'unit', 'cfg', 'K', 'avg_behavior_size', 'support_min', 'support_mean', 'support_max', 'reduction_from_bottom', 'compression_from_bottom']
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

def _deduplicate_executions_by_bottom_iso(*, executions: List[ProcessExecution], types: List[str], hierarchies: Dict[str, ObjectTypeHierarchy], obj_types_map: Dict[str, str], verbose: bool=True) -> Tuple[List[ProcessExecution], Dict[int, List[int]]]:
    bottom_cfg = tuple((0 for _ in types))
    bottom_run = _compute_cfg(bottom_cfg, types, executions, hierarchies, obj_types_map)
    iso_to_indices: Dict[int, List[int]] = defaultdict(list)
    for i, iso_id in enumerate(bottom_run.iso_ids):
        iso_to_indices[iso_id].append(i)
    representatives: List[ProcessExecution] = []
    class_to_original_execs: Dict[int, List[int]] = {}
    for new_idx, original_indices in enumerate(iso_to_indices.values()):
        rep_idx = original_indices[0]
        representatives.append(executions[rep_idx])
        class_to_original_execs[new_idx] = original_indices
    if verbose:
        n_before = len(executions)
        n_after = len(representatives)
        print(f'[dedup] bottom-isomorphism deduplication: {n_before} executions → {n_after} representatives ({n_before - n_after} removed)')
        class_sizes = [len(v) for v in iso_to_indices.values()]
        if class_sizes:
            print(f'[dedup] bottom iso-class sizes: min={min(class_sizes)} mean={sum(class_sizes) / len(class_sizes):.2f} max={max(class_sizes)}')
    return (representatives, class_to_original_execs)

def _level_leq(h: ObjectTypeHierarchy, a: int, b: int) -> bool:
    if a == b:
        return True
    return any((_level_leq(h, up, b) for up in h.cover_up.get(a, [])))

def _cfg_leq(cfg_a: Tuple[int, ...], cfg_b: Tuple[int, ...], types: List[str], hierarchies: Dict[str, ObjectTypeHierarchy]) -> bool:
    return all((_level_leq(hierarchies[tau], cfg_a[i], cfg_b[i]) for i, tau in enumerate(types)))

def run_pipeline(*, ocel_path: str, leading_type: str, s_min: float=0.02, s_max: float=0.8, support_abs: bool=False, top_k: int=10, max_edges: int=4, beam_width: int=12, min_support: int=2, top_k_per_cfg: int=10, exclude_attrs: Sequence[str]=(), out_html: Optional[str]=None, quiet: bool=False, kpis: Optional[List[KPI]]=None, bundle_path: Optional[str]=None, miner: str='subdue') -> Tuple[List[LiftedPattern], RunStats]:
    t0 = time.time()
    timings: Dict[str, float] = {}

    def status(msg: str) -> None:
        print(msg, flush=True)
    if not ocel_path.lower().endswith(('.sqlite', '.db')):
        raise ValueError(f'rho_lift.py now accepts only OCEL-2.0 SQLite logs; got {ocel_path!r}.')
    if not quiet:
        print(f'[load] reading {ocel_path} via pm4py …')
    _t = time.time()
    status(f'[load] reading OCEL from {ocel_path} …')
    ocel = _load_ocel_sqlite_pm4py(ocel_path)
    timings['load_ocel'] = time.time() - _t
    obj_types_map = dict(zip(ocel.objects[OBJ_ID], ocel.objects['ocel:type']))
    types = sorted(set(obj_types_map.values()))
    if not quiet:
        print(f"[exec] extracting executions leading by '{leading_type}' via ocpa …")
    _t = time.time()
    status(f"[exec] extracting executions with leading type '{leading_type}' …")
    executions = _build_process_executions_from_ocpa(ocel, leading_type, ocel_path, verbose=not quiet)
    total_exec_events = sum((len(ex.events) for ex in executions))
    total_exec_edges = sum((len(ex.edges) for ex in executions))
    print(f'[exec] sum events+objects+edges over executions = {total_exec_events + total_exec_edges}')
    timings['execution_extraction'] = time.time() - _t
    if not quiet:
        sizes = [len(ex.events) + len(ex.objects) for ex in executions]
        print(f'[exec] {len(executions)} executions (size min={min(sizes, default=0)} mean={(mean(sizes) if sizes else 0):.1f} max={max(sizes, default=0)})')
    _t = time.time()
    status('[cfg-fn] building abstraction functions …')
    hierarchies, level_names, per_type_attrs = _build_attribute_hierarchies(ocel.objects, obj_types_map, exclude_attrs=exclude_attrs)
    timings['rho_hierarchy_building'] = time.time() - _t
    executions, bottom_iso_groups = _deduplicate_executions_by_bottom_iso(executions=executions, types=types, hierarchies=hierarchies, obj_types_map=obj_types_map, verbose=not quiet)
    if kpis is None:
        kpis = [BUILTIN_KPIS['duration']]
    primary_kpi_name = kpis[0].name
    _t = time.time()
    kpi_values = compute_kpi_matrix(executions, kpis)
    timings['kpi_computation'] = time.time() - _t
    if not quiet:
        print(f'[kpi ] registered: ' + ', '.join((f'{k.name} ({k.unit})' for k in kpis)) + f'  — primary = {primary_kpi_name}')
        for k in kpis:
            vals = kpi_values[k.name]
            if vals:
                print(f'        {k.name:>14s}  min={min(vals):.3f}  mean={mean(vals):.3f}  max={max(vals):.3f}')
    if not quiet:
        _print_functional_dependencies(per_type_attrs, obj_types_map)
    if not quiet:
        print('[cfg-fn] function space per object type:')
        for tau in types:
            names = level_names[tau]
            rs = [f'identity' if n == 'id' else 'map→type' if n == 'type' else f'cfg_{n}' for n in names]
            print(f'        {tau:>10s}  :  ' + '  |  '.join(rs))
    s_abs_min = int(round(s_min * len(executions))) if not support_abs else int(s_min)
    s_abs_max = int(round(s_max * len(executions))) if not support_abs else int(s_max)
    if not quiet:
        print(f"[cfg ] support window: [{s_abs_min}, {s_abs_max}] ({('absolute' if support_abs else 'fraction')})")
    _t = time.time()
    status('[cfg ] enumerating configurations …')
    all_cfgs: List[Tuple[int, ...]] = _enumerate_configs_top_down(types, hierarchies)
    status(f'[cfg ] {len(all_cfgs)} configurations generated')
    timings['configuration_enumeration'] = time.time() - _t
    if not quiet:
        print(f'[cfg ] enumerating {len(all_cfgs)} configurations …')
    bottom_cfg = tuple((0 for _ in types))
    run_bottom = _compute_cfg(bottom_cfg, types, executions, hierarchies, obj_types_map)
    bottom_len = run_bottom.K
    runs: List[CfgRun] = []
    all_runs: List[CfgRun] = []
    runs_by_cfg: Dict[Tuple[int, ...], CfgRun] = {}
    pruned_cfgs: Set[Tuple[int, ...]] = set()
    n_skipped = 0
    _t = time.time()
    total_behavior_time = 0.0
    total_iso_time = 0.0
    if not quiet:
        print(f'[mine] miner = {miner!r}')
    all_scored: List[LiftedPattern] = []
    bundle_by_cfg: Dict[Tuple[int, ...], List[BundlePattern]] = {}
    want_bundle = bundle_path is not None
    mining_stats_by_cfg: Dict[Tuple[int, ...], Dict[str, Any]] = {}
    lattice_metrics: Dict[Tuple, LatticeNodeMetrics] = {}
    total_mining_time = 0.0
    n_mined = 0

    def _stash_lattice_metrics(r: CfgRun) -> None:
        n = len(r.behaviors)
        K = len(set(r.iso_ids))
        s = sum((_behavior_size(b) for b in r.behaviors)) / n if n else 0.0
        if K:
            counts = Counter(r.iso_ids)
            sup_min = min(counts.values())
            sup_max = max(counts.values())
            sup_mean = n / K
        else:
            sup_min = sup_max = 0
            sup_mean = 0.0
        lattice_metrics[r.cfg] = LatticeNodeMetrics(K=K, s=s, sup_min=sup_min, sup_max=sup_max, sup_mean=sup_mean)

    def _stream_mine(r: CfgRun) -> None:
        nonlocal total_mining_time, n_mined
        n_mined += 1
        m_cfg = lattice_metrics[r.cfg]
        red_cfg = behavior_reduction(lattice_metrics, bottom_cfg, r.cfg)
        comp_cfg = behavior_compression(lattice_metrics, bottom_cfg, r.cfg)
        status(f'[mine] configuration {n_mined}')
        print(f'[mine] cfg {n_mined}  Φ = {_fmt_cfg(r.cfg, types, level_names)}  (K={r.K}/{r.n_executions}, s={m_cfg.s:.2f}, reduction_from_bottom={red_cfg:+.4f} {_pct(red_cfg)}, behavior_compression={comp_cfg:+.4f} {_pct(comp_cfg)})')
        t_mine = time.time()
        scored, bundled, mining_stats = _mine_and_score(r, kpi_values=kpi_values, primary_kpi=primary_kpi_name, s_min_abs=s_abs_min, s_max_abs=s_abs_max, max_edges=max_edges, beam_width=beam_width, min_support=min_support, top_k_per_cfg=top_k_per_cfg, verbose=not quiet, collect_all=want_bundle, miner=miner)
        total_mining_time += time.time() - t_mine
        mining_stats_by_cfg[r.cfg] = mining_stats
        if not quiet:
            print(f'        → {len(scored)} lifted patterns kept from this cfg  (top-{top_k_per_cfg} by KPI)')
        all_scored.extend(scored)
        if want_bundle:
            bundle_by_cfg[r.cfg] = bundled
        r.behaviors = []
        r.iso_ids = []
    _stash_lattice_metrics(run_bottom)
    status('[cfg ] computing behaviors and isomorphism classes …')
    for j, cfg in enumerate(all_cfgs):
        if cfg != bottom_cfg and any((_cfg_leq(cfg, p, types, hierarchies) for p in pruned_cfgs)):
            n_skipped += 1
            if not quiet:
                print(f'[skip] sotto configurazione potata: {cfg}')
            continue
        if cfg == bottom_cfg:
            run = run_bottom
        else:
            run = _compute_cfg(cfg, types, executions, hierarchies, obj_types_map)
        class_sizes = Counter(run.iso_ids)
        behavior_sizes = [_behavior_size(b) for b in run.behaviors]
        avg_behavior_size = sum(behavior_sizes) / len(behavior_sizes) if behavior_sizes else 0.0
        cfg_height = _cfg_height(cfg, types, hierarchies)
        print(f'[iso ] cfg={cfg}  h={cfg_height}  K={run.K}/{run.n_executions}  s={avg_behavior_size:.2f}  class sizes min={min(class_sizes.values())} mean={run.n_executions / run.K:.2f} max={max(class_sizes.values())}')
        total_behavior_time += run.behavior_time_s
        total_iso_time += run.iso_time_s
        runs_by_cfg[cfg] = run
        if cfg != bottom_cfg and run.K == bottom_len:
            pruned_cfgs.add(cfg)
            if not quiet:
                print(f'[prune] K = n_executions ({run.K}); sub-lattice below this cfg is skipped\n        Φ = {_fmt_cfg(cfg, types, level_names)}')
        ok, why = _is_interesting(run)
        run.is_interesting = ok
        run.reason_skipped = why
        all_runs.append(run)
        _stash_lattice_metrics(run)
        if ok:
            runs.append(run)
            if cfg != bottom_cfg:
                _stream_mine(run)
        else:
            n_skipped += 1
            if cfg != bottom_cfg:
                run.behaviors = []
                run.iso_ids = []
        if not quiet and (j + 1) % max(1, len(all_cfgs) // 10) == 0:
            print(f'        processed {j + 1}/{len(all_cfgs)} ({len(runs)} interesting)')
        if (j + 1) % max(1, len(all_cfgs) // 10) == 0:
            status(f'[cfg ] processed {j + 1}/{len(all_cfgs)} configurations ({len(runs)} interesting)')
    timings['behavior_computation'] = total_behavior_time
    timings['isomorphism_check'] = total_iso_time
    timings['behavior_and_iso_computation'] = time.time() - _t
    timings['lattice_metrics'] = 0.0
    mine_runs: List[CfgRun] = [run for run in all_runs if run.is_interesting or run.cfg == bottom_cfg]
    n_metric_skipped = 0
    if not quiet:
        print(f'[cfg ] {len(runs)} interesting / {len(all_cfgs)} total')
        print(f'[cfg ] {len(mine_runs)} mineable')
    if not quiet:
        if bottom_cfg not in lattice_metrics:
            print('[latt] bottom Φ was pruned; skipping bottom-based lattice report')
        else:
            mb = lattice_metrics[bottom_cfg]
            print('[latt] reduction/compression measured from bottom Φ')
            print(f'bottom cfg = {_fmt_cfg(bottom_cfg, types, level_names)}')
            print(f'bottom abs = {mb.K}, bottom s = {mb.s:.2f}')
            print('[latt] per-configuration reduction/compression from bottom:')
            for run in all_runs:
                m_cfg = lattice_metrics[run.cfg]
                red_cfg = behavior_reduction(lattice_metrics, bottom_cfg, run.cfg)
                comp_cfg = behavior_compression(lattice_metrics, bottom_cfg, run.cfg)
                print(f'cfg = {_fmt_cfg(run.cfg, types, level_names)}  K={run.K}/{run.n_executions}, s={m_cfg.s:.2f}, reduction_from_bottom={red_cfg:+.4f} {_pct(red_cfg)}, behavior_compression={comp_cfg:+.4f} {_pct(comp_cfg)}, support=[{m_cfg.sup_min}, {m_cfg.sup_mean:.2f}, {m_cfg.sup_max}], interesting={run.is_interesting}')
    status(f'[mine] finalising deferred mining ({len(runs)} streaming + bottom Φ at end) …')
    for run in mine_runs:
        if run.cfg in mining_stats_by_cfg:
            continue
        _stream_mine(run)
    timings['subgraph_mining_and_kpi_scoring'] = total_mining_time
    all_scored.sort(key=lambda p: -p.delta)
    top = all_scored[:top_k]
    dataset_name = Path(ocel_path).stem
    max_cfg_height = max((_cfg_height(run.cfg, types, hierarchies) for run in all_runs), default=1)
    timing_rows: List[CfgTimingRow] = []
    p = Path(ocel_path)
    for run in all_runs:
        m_cfg = lattice_metrics[run.cfg]
        h = _cfg_height(run.cfg, types, hierarchies)
        h_norm = h / max_cfg_height if max_cfg_height else 0.0
        k_ratio = run.K / run.n_executions if run.n_executions else 0.0
        red_cfg = behavior_reduction(lattice_metrics, bottom_cfg, run.cfg)
        comp_cfg = behavior_compression(lattice_metrics, bottom_cfg, run.cfg)
        mining_stats = mining_stats_by_cfg.get(run.cfg, {})
        timing_rows.append(CfgTimingRow(dataset=dataset_name, cfg=repr(run.cfg), phi=_fmt_cfg(run.cfg, types, level_names), interesting=run.is_interesting, reason_skipped=run.reason_skipped, K=run.K, n_executions=run.n_executions, K_ratio=k_ratio, cfg_height=h, cfg_height_norm=h_norm, avg_behavior_size=m_cfg.s, support_min=m_cfg.sup_min, support_mean=m_cfg.sup_mean, support_max=m_cfg.sup_max, reduction_from_bottom=red_cfg, compression_from_bottom=comp_cfg, behavior_time_s=run.behavior_time_s, mining_time_s=mining_stats.get('mining_time_s'), miner=mining_stats.get('miner', ''), n_graphs=mining_stats.get('n_graphs', 0), avg_mining_graph_size=mining_stats.get('avg_mining_graph_size', 0.0), total_mining_graph_size=mining_stats.get('total_mining_graph_size', 0), min_support_effective=mining_stats.get('min_support_effective', 0), n_patterns_raw=mining_stats.get('n_patterns_raw', 0), n_patterns_in_window=mining_stats.get('n_patterns_in_window', 0), pattern_support_min=mining_stats.get('pattern_support_min', 0), pattern_support_mean=mining_stats.get('pattern_support_mean', 0.0), pattern_support_median=mining_stats.get('pattern_support_median', 0.0), pattern_support_max=mining_stats.get('pattern_support_max', 0), pattern_support_ratio_mean=mining_stats.get('pattern_support_ratio_mean', 0.0), pattern_support_ratio_median=mining_stats.get('pattern_support_ratio_median', 0.0), pattern_support_ratio_max=mining_stats.get('pattern_support_ratio_max', 0.0), high_support_threshold_ratio=mining_stats.get('high_support_threshold_ratio', 0.25), high_support_threshold_abs=mining_stats.get('high_support_threshold_abs', 0), n_high_support_patterns=mining_stats.get('n_high_support_patterns', 0), high_support_pattern_ratio=mining_stats.get('high_support_pattern_ratio', 0.0)))
    cfg_timing_csv = str(p.with_name(p.stem + '_cfg_timing.csv'))
    save_cfg_timing_csv(cfg_timing_csv, timing_rows)
    if not quiet:
        print(f'[csv ] per-configuration timings written to {cfg_timing_csv}')
    stats = RunStats(n_events=len(ocel.events), n_objects=len(ocel.objects), n_executions=len(executions), types=types, n_cfg_total=len(all_cfgs), n_cfg_interesting=len(runs), n_cfg_skipped=n_skipped + n_metric_skipped, n_patterns_total=len(all_scored), n_patterns_feas=len(top))
    elapsed = time.time() - t0
    top_cfg = _top_cfg(types, hierarchies)
    p = Path(ocel_path)
    stats_csv = str(p.with_suffix('.csv'))
    save_run_stats_csv(stats_csv, timings=timings, elapsed=elapsed, types=types, level_names=level_names, lattice_metrics=lattice_metrics, bottom_cfg=bottom_cfg, top_cfg=top_cfg, n_events=stats.n_events, n_objects=stats.n_objects, n_executions=stats.n_executions, n_cfg_total=stats.n_cfg_total, n_cfg_interesting=stats.n_cfg_interesting, n_cfg_mineable=len(mine_runs), n_cfg_skipped=stats.n_cfg_skipped, n_patterns_total=len(all_scored), n_patterns_top=len(top))
    if out_html:
        if not quiet:
            print(f'[html] rendering {len(top)} patterns → {out_html}')
        max_delta = max((p.delta for p in top), default=1.0) or 1.0
        items = []
        for pat in top:
            t_heat = min(1.0, pat.delta / max_delta)
            items.append({'pattern': pat.pattern, 'size_nodes': pat.size_nodes, 'size_edges': pat.size_edges, 'support': pat.support, 'n_graphs': pat.n_graphs, 'in_exec_idx': pat.in_exec_idx, 'out_exec_idx': pat.out_exec_idx, 'mu_in': pat.mu_in, 'mu_out': pat.mu_out, 'delta': pat.delta, 'signed_delta': pat.signed_delta, 'mu_in_h': _fmt_duration_seconds(pat.mu_in), 'mu_out_h': _fmt_duration_seconds(pat.mu_out), 'delta_h': _fmt_duration_seconds(pat.delta), 'signed_delta_h_abs': _fmt_duration_seconds(abs(pat.signed_delta)), 'phi_str': _fmt_cfg(pat.cfg, types, level_names), 'accent': column_accent(round(10 * t_heat), 10)})
        render_html(ocel_path=ocel_path, leading_type=leading_type, n_executions=len(executions), n_cfg_eval=len(all_cfgs), n_cfg_interesting=len(runs), s_min=s_abs_min, s_max=s_abs_max, kpi_label='execution duration (max_ts − min_ts)', elapsed_s=elapsed, patterns=items, out_path=out_html, k=len(items))
    if bundle_path:
        if not quiet:
            print(f'[bund] packaging full run → {bundle_path}')
        bundle_runs: List[BundleCfgRun] = []
        n_patterns_total = 0
        for run in all_runs:
            pats = bundle_by_cfg.get(run.cfg, [])
            n_patterns_total += len(pats)
            m_cfg = lattice_metrics[run.cfg]
            bundle_runs.append(BundleCfgRun(cfg=run.cfg, K=run.K, n_executions=run.n_executions, is_interesting=run.is_interesting, reason_skipped=run.reason_skipped, iso_ids=list(run.iso_ids), reduction_from_bottom=behavior_reduction(lattice_metrics, bottom_cfg, run.cfg), compression_from_bottom=behavior_compression(lattice_metrics, bottom_cfg, run.cfg), avg_behavior_size=m_cfg.s, patterns=pats))
        bundle = RunBundle(ocel_path=ocel_path, leading_type=leading_type, timestamp_iso=_dt.datetime.utcnow().isoformat() + 'Z', elapsed_s=elapsed, args={'s_min': s_min, 's_max': s_max, 'support_abs': support_abs, 'top_k': top_k, 'max_edges': max_edges, 'beam_width': beam_width, 'min_support': min_support, 'top_k_per_cfg': top_k_per_cfg, 'exclude_attrs': list(exclude_attrs), 's_abs_min': s_abs_min, 's_abs_max': s_abs_max, 'miner': miner, 'bottom_cfg': bottom_cfg}, types=types, obj_types_map=obj_types_map, level_names=level_names, per_type_attrs=per_type_attrs, executions=executions, kpi_specs=[{'name': k.name, 'unit': k.unit, 'description': k.description} for k in kpis], kpi_values=kpi_values, primary_kpi=primary_kpi_name, cfg_runs=bundle_runs, n_events=stats.n_events, n_objects=stats.n_objects, n_cfg_total=stats.n_cfg_total, n_cfg_interesting=stats.n_cfg_interesting, n_cfg_skipped=stats.n_cfg_skipped, n_patterns_total=n_patterns_total)
        save_bundle(bundle, bundle_path)
        if not quiet:
            print(f'[bund] wrote {n_patterns_total} patterns across {len(bundle_runs)} Φ to {bundle_path}')
    cfg_csv = str(p.with_name(p.stem + '_configs.csv'))
    save_cfg_metrics_csv(cfg_csv, runs=all_runs, types=types, level_names=level_names, lattice_metrics=lattice_metrics, bottom_cfg=bottom_cfg)
    if not quiet:
        print(f'[csv] configuration metrics written to {cfg_csv}')
    status(f'[done] elapsed {elapsed:.1f}s — {len(top)}/{len(all_scored)} patterns kept')
    return (top, stats)

def _print_top(top: List[LiftedPattern], types: List[str], level_names: Dict[str, List[str]]) -> None:
    bar = '─' * 78
    print(bar)
    print(f'TOP {len(top)} KPI patterns')
    print(bar)
    for i, p in enumerate(top, 1):
        sign = '+' if p.signed_delta >= 0 else '−'
        print(f'  {i:2d}. KPI = {_fmt_duration_seconds(p.delta):>10s}  (signed {sign}{_fmt_duration_seconds(abs(p.signed_delta))})')
        print(f'      support = {p.support}/{p.n_graphs}  |E_in|={len(p.in_exec_idx)}  |E_out|={len(p.out_exec_idx)}')
        print(f'      min  = {_fmt_duration_seconds(p.mu_in)}     max = {_fmt_duration_seconds(p.mu_out)}')
        print(f'      size  = {p.size_nodes} nodes, {p.size_edges} edges')
        print(f'      cfg : {_fmt_cfg(p.cfg, types, level_names)}')
    print(bar)

def main(argv: Optional[List[str]]=None) -> int:
    ap = argparse.ArgumentParser(description='configured SUBDUE mining with KPI-lift scoring. Input: OCEL-2.0 SQLite only.')
    ap.add_argument('ocel', help='Path to an .sqlite OCEL-2.0 log.')
    ap.add_argument('--leading', required=True, help='Object type used as leading object for execution extraction (delegated to ocpa).')
    ap.add_argument('--s-min', type=float, default=0.02, help='Minimum pattern support (default: 0.02, fraction).')
    ap.add_argument('--s-max', type=float, default=0.8, help='Maximum pattern support (default: 0.80, fraction).')
    ap.add_argument('--support-abs', action='store_true', help='Interpret --s-min/--s-max as absolute counts instead of fractions.')
    ap.add_argument('--top-k', type=int, default=10, help='Number of top patterns to keep globally.')
    ap.add_argument('--top-k-per-cfg', type=int, default=10, help='Number of patterns SUBDUE returns per Φ before lift scoring (default: 10).')
    ap.add_argument('--max-edges', type=int, default=4, help='Maximum pattern size (edges) for SUBDUE.')
    ap.add_argument('--beam', type=int, default=12, help='Beam width for SUBDUE expansion.')
    ap.add_argument('--min-support', type=int, default=2, help="Miner's internal minimum support (integer).")
    ap.add_argument('--miner', choices=['subdue', 'gspan'], default='subdue', help="Subgraph miner to use.  'subdue' is the beam-search built-in; 'gspan' delegates to the gspan-mining PyPI package via a gadget encoding that preserves label-set subset matching (pip install gspan-mining required).  Default: subdue.")
    ap.add_argument('--exclude-attrs', default='', help='Comma-separated attribute names to exclude from the configuration space.')
    ap.add_argument('--out', default=None, help='Output HTML path. Default: rho_lift_report.html inside the input OCEL directory. Pass empty string to skip.')
    ap.add_argument('--bundle', default=None, help='If set, save the full run (all Φ, all patterns, all KPIs) to this .pkl file for the interactive explorer (rho_explorer.py).')
    ap.add_argument('--kpi', action='append', default=None, help="KPI(s) to compute. Can be passed multiple times. Built-ins: duration, n_events, n_objects, n_activities, event_density. Custom: 'name:=EXPR' or just 'EXPR'. The first --kpi is the PRIMARY KPI used for KPI ranking (default: duration).")
    ap.add_argument('--quiet', action='store_true', help='Silence progress logs.')
    args = ap.parse_args(argv)
    excl = [a.strip() for a in args.exclude_attrs.split(',') if a.strip()]
    kpis = resolve_kpi_flags(args.kpi or [])
    if args.out is None:
        out_h = str(Path(args.ocel).parent / 'rho_lift_report.html')
    elif args.out.strip():
        out_h = args.out
    else:
        out_h = None
    top, stats = run_pipeline(ocel_path=args.ocel, leading_type=args.leading, s_min=args.s_min, s_max=args.s_max, support_abs=args.support_abs, top_k=args.top_k, max_edges=args.max_edges, beam_width=args.beam, min_support=args.min_support, top_k_per_cfg=args.top_k_per_cfg, exclude_attrs=excl, out_html=out_h, quiet=args.quiet, kpis=kpis, bundle_path=args.bundle, miner=args.miner)
    ocel = _load_ocel_sqlite_pm4py(args.ocel)
    obj_types_map = dict(zip(ocel.objects[OBJ_ID], ocel.objects['ocel:type']))
    types = sorted(set(obj_types_map.values()))
    _, level_names, _ = _build_attribute_hierarchies(ocel.objects, obj_types_map, exclude_attrs=excl)
    _print_top(top, types, level_names)
    if out_h:
        print(f'  HTML report written to {out_h}')
    if args.bundle:
        print(f'  Interactive bundle saved to {args.bundle}')
        print(f'  → open it with:  python3 rho_explorer.py {args.bundle}')
    return 0
if __name__ == '__main__':
    sys.exit(main())
