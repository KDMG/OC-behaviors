from __future__ import annotations

import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _ROOT not in _sys.path:
    _sys.path.insert(0, _ROOT)

import argparse
import math
import os
from typing import Any, Dict, List

from flask import Flask, abort, jsonify, request, send_from_directory

from explorer.oc_state import ExplorerState, fmt_cfg
from explorer.oc_attrs import attribute_breakdown
from explorer.oc_tree_analysis import (
    analyze_all_cfgs,
    compute_leaf_membership,
    cross_leaf_distribution,
    behavior_relation as tree_behavior_relation,
    summarize_cfg_tree,
)

def _kpi_unit_map(state: ExplorerState) -> Dict[str, str]:
    return {k.get("name"): k.get("unit", "") for k in state.bundle.kpi_specs}

def _fmt_kpi(value: float, unit: str) -> Dict[str, Any]:
    pretty = f"{value:.3f}"
    if unit in {"s", "second", "seconds"}:
        if value >= 86_400:
            pretty = f"{value / 86_400:.2f} d"
        elif value >= 3_600:
            pretty = f"{value / 3_600:.2f} h"
        elif value >= 60:
            pretty = f"{value / 60:.2f} min"
        else:
            pretty = f"{value:.2f} s"
    elif unit:
        pretty = f"{value:.3f} {unit}"
    return {"raw": float(value), "unit": unit, "pretty": pretty}

def _behavior_to_cy(behavior) -> Dict[str, Any]:
    nodes = [
        {"data": {"id": f"n{n}", "label": d.get("label", str(n))}}
        for n, d in behavior.nodes(data=True)
    ]
    edges: List[Dict[str, Any]] = []
    eid = 0
    for u, v, d in behavior.edges(data=True):
        labels = d.get("labels")
        if labels is None:
            lbl = d.get("label")
            sorted_lbls = [str(lbl)] if lbl is not None else [""]
        else:
            sorted_lbls = sorted(str(l) for l in labels)
        n_lbls = len(sorted_lbls)
        for k, lbl_text in enumerate(sorted_lbls):
            edges.append({"data": {
                "id": f"e{eid}",
                "source": f"n{u}",
                "target": f"n{v}",
                "label": lbl_text,
                "parallel_idx": k,
                "parallel_n": n_lbls,
            }})
            eid += 1
    return {"nodes": nodes, "edges": edges}

def _quantile(sorted_vals: List[float], q: float) -> float:
    if not sorted_vals:
        return 0.0
    n = len(sorted_vals)
    if n == 1:
        return float(sorted_vals[0])
    pos = (n - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return float(sorted_vals[lo])
    frac = pos - lo
    return float(sorted_vals[lo]) * (1 - frac) + float(sorted_vals[hi]) * frac

def _summary_stats(values: List[float]) -> Dict[str, float]:
    if not values:
        return {"n": 0, "min": 0.0, "max": 0.0, "mean": 0.0,
                "median": 0.0, "q25": 0.0, "q75": 0.0, "std": 0.0}
    s = sorted(values)
    m = sum(values) / len(values)
    var = sum((v - m) ** 2 for v in values) / len(values)
    return {
        "n": len(values),
        "min": float(s[0]),
        "max": float(s[-1]),
        "mean": float(m),
        "median": _quantile(s, 0.5),
        "q25": _quantile(s, 0.25),
        "q75": _quantile(s, 0.75),
        "std": float(math.sqrt(var)),
    }

def find_min_occupied_leaf_target_cfgs(
        state,
        source_cfg_idx: int,
        source_leaf_id: int,
        kpi = None,
        target_cfgs = None,
        top_n = None,
        max_depth: int = 6,
        min_samples_leaf: int = 20,
        cache = None,
) -> Dict[str, Any]:

    res = cross_leaf_distribution(
        state=state,
        source_cfg_idx=source_cfg_idx,
        source_leaf_id=source_leaf_id,
        kpi=kpi,
        target_cfgs=target_cfgs,
        top_n=None,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        cache=cache,
    )

    if res.get("status") != "ok":
        return {
            **res,
            "ranking_goal": "minimize_occupied_target_leaves",
            "best_target": None,
        }

    ranked_targets: List[Dict[str, Any]] = []

    for tgt in res.get("targets", []):
        leaves = tgt.get("leaves", [])

        if not leaves:
            continue

        occupied_leaf_count = len(leaves)

        top_leaf = max(leaves, key=lambda lf: lf.get("n_overlap", 0))

        top_overlap = int(top_leaf.get("n_overlap", 0))
        top_frac_of_source = float(top_leaf.get("frac_of_source", 0.0) or 0.0)
        top_frac_of_target_leaf = float(
            top_leaf.get("frac_of_target_leaf", 0.0) or 0.0
        )

        ranked_targets.append({
            **tgt,

            "occupied_leaf_count": occupied_leaf_count,

            "n_occupied_target_leaves": occupied_leaf_count,

            "top_leaf_id": top_leaf.get("leaf_id"),
            "top_overlap": top_overlap,
            "top_frac_of_source": top_frac_of_source,
            "top_frac_of_target_leaf": top_frac_of_target_leaf,
        })

    ranked_targets.sort(
        key=lambda r: (
            r["occupied_leaf_count"],
            -r["top_overlap"],
            r.get("entropy", 0.0),
            -r["top_frac_of_source"],
        )
    )

    if top_n is not None:
        ranked_targets = ranked_targets[:top_n]

    best_target = ranked_targets[0] if ranked_targets else None

    return {
        **res,
        "ranking_goal": "minimize_occupied_target_leaves",
        "best_target": best_target,
        "targets": ranked_targets,
    }

def _level_name(level_names: Dict[str, Any], tau: str, L: int) -> str:
    names = level_names.get(tau, None)

    if isinstance(names, dict):
        return str(names.get(L, L))

    if isinstance(names, list):
        if 0 <= L < len(names):
            return str(names[L])
        return str(L)

    return str(L)

def print_rho_dependencies(state: ExplorerState) -> None:
    b = state.bundle

    for tau in b.types:
        h = state.hierarchies.get(tau)
        if h is None:
            continue

def create_app(state: ExplorerState) -> Flask:
    here = os.path.dirname(os.path.abspath(__file__))
    app = Flask(
        __name__,
        static_folder=os.path.join(here, "oc_static"),
        static_url_path="/static",
        template_folder=os.path.join(here, "oc_templates"),
    )
    kpi_unit_of = _kpi_unit_map(state)

    @app.get("/")
    def index():
        return send_from_directory(
            os.path.join(here, "oc_templates"), "index.html",
        )

    @app.get("/api/summary")
    def api_summary():
        b = state.bundle
        return jsonify({
            "ocel_path": b.ocel_path,
            "leading_type": b.leading_type,
            "timestamp": b.timestamp_iso,
            "elapsed_s": b.elapsed_s,
            "types": b.types,
            "n_events": b.n_events,
            "n_objects": b.n_objects,
            "n_executions": len(b.executions),
            "n_cfg_total": b.n_cfg_total,
            "n_cfg_interesting": b.n_cfg_interesting,
            "n_cfg_skipped": b.n_cfg_skipped,
            "n_behaviors_total": getattr(b, "n_behaviors_total", getattr(b, "n_behaviors_total", 0)),
            "kpis": b.kpi_specs,
            "primary_kpi": b.primary_kpi,
            "level_names": b.level_names,
        })

    @app.get("/api/lattice")
    def api_lattice():
        b = state.bundle
        level_covers: Dict[str, Dict[int, List[int]]] = {}
        level_covered_by: Dict[str, Dict[int, List[int]]] = {}
        for tau in b.types:
            h = state.hierarchies.get(tau)
            if h is None:
                continue
            level_covers[tau] = {
                L: list(h.cover_up.get(L, [])) for L in range(h.k + 1)
            }
            level_covered_by[tau] = {
                L: list(h.cover_dn.get(L, [])) for L in range(h.k + 1)
            }

        kept_counts = {
            r.cfg_idx: len(state.behaviors_after_subsumption(r.cfg_idx))
            for r in state.lattice
        }
        rows = [
            {
                "cfg_idx":              r.cfg_idx,
                "cfg":                  r.cfg,
                "cfg_str":              r.cfg_str,
                "K":                    r.K,
                "n_abstractions":       r.n_abstractions,
                "n_abstractions_kept":  kept_counts.get(r.cfg_idx, r.n_abstractions),
                "n_executions":         r.n_executions,
                "is_interesting":       r.is_interesting,
                "best_delta":           r.best_delta,
            }
            for r in state.lattice
        ]
        return jsonify({
            "types": b.types,
            "level_names": b.level_names,
            "level_covers": level_covers,
            "level_covered_by": level_covered_by,
            "rows": rows,
            "n_executions": len(b.executions),
            "primary_kpi": b.primary_kpi,
        })

    @app.get("/api/tree/cfg/<int:cfg_idx>/leaf/<int:leaf_id>/compact-cross")
    def api_compact_cross(cfg_idx: int, leaf_id: int):
        if not (0 <= cfg_idx < len(state.bundle.cfg_runs)):
            abort(404)

        kpi = request.args.get("kpi", state.bundle.primary_kpi)
        if kpi not in state.bundle.kpi_values:
            kpi = state.bundle.primary_kpi

        top_n = request.args.get("top_n", type=int)
        target_cfg = request.args.get("target_cfg", type=int)

        if target_cfg is not None:
            if not (0 <= target_cfg < len(state.bundle.cfg_runs)):
                return jsonify({
                    "status": "bad_target_cfg",
                    "target_cfg": target_cfg,
                    "n_cfg_runs": len(state.bundle.cfg_runs),
                }), 400

        target_cfgs = [target_cfg] if target_cfg is not None else None

        cache = getattr(state, "_leaves_cache", None)
        if cache is None:
            cache = {}
            state._leaves_cache = cache

        payload = find_min_occupied_leaf_target_cfgs(
            state=state,
            source_cfg_idx=cfg_idx,
            source_leaf_id=leaf_id,
            kpi=kpi,
            target_cfgs=target_cfgs,
            top_n=top_n,
            max_depth=3,
            min_samples_leaf=20,
            cache=cache,
        )

        return jsonify(payload)

    @app.get("/api/cfg/<int:ci>")
    def api_cfg(ci: int):
        if not (0 <= ci < len(state.bundle.cfg_runs)):
            abort(404)
        b = state.bundle
        run = b.cfg_runs[ci]
        kpi = request.args.get("kpi", b.primary_kpi)
        if kpi not in b.kpi_values:
            kpi = b.primary_kpi
        unit = kpi_unit_of.get(kpi, "")

        kept_behaviors = state.behaviors_after_subsumption(ci)
        rows = []
        for pi, p in kept_behaviors:
            stats = p.kpi_stats.get(kpi, {})
            delta = stats.get("delta", 0.0)
            signed = stats.get("signed_delta", delta)
            rows.append({
                "cfg_idx": ci,
                "behavior_idx": pi,
                "size_nodes": p.size_nodes,
                "size_edges": p.size_edges,
                "support": p.support,
                "n_in": len(p.in_exec_idx),
                "n_out": len(p.out_exec_idx),
                "delta": delta,
                "signed_delta": signed,
                "delta_fmt": _fmt_kpi(abs(delta), unit),
                "mu_in":  stats.get("mu_in", 0.0),
                "mu_out": stats.get("mu_out", 0.0),
                "mu_in_fmt":  _fmt_kpi(stats.get("mu_in", 0.0), unit),
                "mu_out_fmt": _fmt_kpi(stats.get("mu_out", 0.0), unit),
            })
        rows.sort(key=lambda r: abs(r["delta"]), reverse=True)
        return jsonify({
            "cfg_idx": ci,
            "cfg": list(run.cfg),
            "cfg_str": fmt_cfg(b.types, b.level_names, run.cfg),
            "K": run.K,
            "n_abstractions_raw": len(run.patterns),
            "n_abstractions_kept": len(rows),
            "n_executions": run.n_executions,
            "kpi": kpi,
            "kpi_unit": unit,
            "rows": rows,
        })

    @app.get("/api/cfg/<int:ci>/situation")
    def api_cfg_situation(ci: int):
        if not (0 <= ci < len(state.bundle.cfg_runs)):
            abort(404)
        b = state.bundle
        run = b.cfg_runs[ci]

        kept_behaviors = state.behaviors_after_subsumption(ci)

        kpi_meta = [{"name": k.get("name"), "unit": k.get("unit", "")}
                    for k in b.kpi_specs]

        behavior_meta = []
        in_sets = []
        primary = b.primary_kpi
        for pi, p in kept_behaviors:
            stats = p.kpi_stats.get(primary, {})
            behavior_meta.append({
                "behavior_idx": pi,
                "support": p.support,
                "size_nodes": p.size_nodes,
                "size_edges": p.size_edges,
                "delta": stats.get("delta", 0.0),
                "signed_delta": stats.get("signed_delta", 0.0),
            })
            in_sets.append(frozenset(p.in_exec_idx))

        rows = []
        for exec_i, ex in enumerate(b.executions):
            row = {"exec_idx": exec_i, "n_events": len(ex.events)}
            for kc in kpi_meta:
                vals = b.kpi_values.get(kc["name"], [])
                row[kc["name"]] = vals[exec_i] if exec_i < len(vals) else None
            for bc, in_set in zip(behavior_meta, in_sets):
                row[f"b{bc['behavior_idx']}"] = exec_i in in_set
            rows.append(row)

        return jsonify({
            "cfg_idx": ci,
            "cfg_str": fmt_cfg(b.types, b.level_names, run.cfg),
            "n_executions": run.n_executions,
            "kpi_meta": kpi_meta,
            "behavior_meta": behavior_meta,
            "rows": rows,
        })

    @app.get("/api/cfg/<int:cfg_idx>/leaves")
    def api_cfg_leaves(cfg_idx: int):
        b = state.bundle

        if not (0 <= cfg_idx < len(b.cfg_runs)):
            abort(404, description="configuration index out of range")

        run = b.cfg_runs[cfg_idx]

        leaves = {}
        for exec_i, leaf_id in enumerate(run.iso_ids):
            leaves.setdefault(int(leaf_id), []).append(exec_i)

        rows = [
            {
                "leaf_id": leaf_id,
                "n_executions": len(execs),
                "exec_idx": execs,
            }
            for leaf_id, execs in sorted(leaves.items())
        ]

        return jsonify({
            "cfg_idx": cfg_idx,
            "cfg": list(run.cfg),
            "cfg_str": fmt_cfg(b.types, b.level_names, run.cfg),
            "K": run.K,
            "n_leaves": len(rows),
            "leaves": rows,
        })

    @app.get("/api/tree/all")
    def api_tree_all():

        def _behavior_to_text(behavior) -> Dict[str, Any]:
            nodes = []

            for n, d in behavior.nodes(data=True):
                nodes.append({
                    "id": int(n) if isinstance(n, int) else str(n),
                    "label": d.get("label", str(n)),
                })

            edges = []

            for u, v, d in behavior.edges(data=True):
                labels = d.get("labels")

                if labels is None:
                    lbl = d.get("label")
                    sorted_lbls = [str(lbl)] if lbl is not None else [""]
                else:
                    sorted_lbls = sorted(str(l) for l in labels)

                source_label = behavior.nodes[u].get("label", str(u))
                target_label = behavior.nodes[v].get("label", str(v))

                for lbl in sorted_lbls:
                    edges.append({
                        "source": int(u) if isinstance(u, int) else str(u),
                        "target": int(v) if isinstance(v, int) else str(v),
                        "label": lbl,
                        "text": f"{source_label} --{lbl}--> {target_label}",
                    })

            return {
                "nodes": nodes,
                "edges": edges,
                "text": " ; ".join(e["text"] for e in edges),
            }

        b = state.bundle

        kpi = request.args.get("kpi", b.primary_kpi)
        if kpi not in b.kpi_values:
            kpi = b.primary_kpi

        only_interesting = bool(request.args.get("only_interesting", type=int, default=1))
        include_leaves = bool(request.args.get("include_leaves", type=int, default=1))
        include_patterns = bool(request.args.get("include_behaviors", type=int, default=1))
        include_pattern_graphs = bool(request.args.get("include_behavior_graphs", type=int, default=0))

        cache = getattr(state, "_leaves_cache", None)
        if cache is None:
            cache = {}
            state._leaves_cache = cache

        rows = []

        for ci, run in enumerate(b.cfg_runs):
            if only_interesting and not run.is_interesting:
                continue

            kept = state.behaviors_after_subsumption(ci)
            n_kept = len(kept) if kept is not None else len(run.patterns)

            if n_kept < 1:
                continue

            tree_payload = summarize_cfg_tree(state, ci, kpi=kpi)

            row = {
                "cfg_idx": ci,
                "cfg": list(run.cfg),
                "cfg_str": fmt_cfg(b.types, b.level_names, run.cfg),
                "K": run.K,
                "n_abstractions_raw": len(run.patterns),
                "n_abstractions_kept": n_kept,
                "n_executions": run.n_executions,
                "tree": tree_payload,
            }

            if include_leaves:
                key = (ci, kpi)
                leaves_payload = cache.get(key)

                if leaves_payload is None:
                    leaves_payload = compute_leaf_membership(state, ci, kpi=kpi)
                    cache[key] = leaves_payload

                row["n_leaves"] = len(leaves_payload.get("leaves", []))
                row["leaf_membership"] = leaves_payload.get("leaves", [])

            if include_patterns:
                pattern_rows = []

                for pi, p in kept:
                    stats = p.kpi_stats.get(kpi, {})
                    delta = stats.get("delta", 0.0)
                    signed = stats.get("signed_delta", delta)
                    unit = kpi_unit_of.get(kpi, "")

                    pattern_row = {
                        "cfg_idx": ci,
                        "behavior_idx": pi,
                        "pattern_name": f"p{pi}",

                        "size_nodes": p.size_nodes,
                        "size_edges": p.size_edges,
                        "support": p.support,

                        "n_in": len(p.in_exec_idx),
                        "n_out": len(p.out_exec_idx),
                        "in_exec_idx": list(p.in_exec_idx),
                        "out_exec_idx": list(p.out_exec_idx),

                        "delta": delta,
                        "signed_delta": signed,
                        "delta_fmt": _fmt_kpi(abs(delta), unit),

                        "mu_in": stats.get("mu_in", 0.0),
                        "mu_out": stats.get("mu_out", 0.0),
                        "mu_in_fmt": _fmt_kpi(stats.get("mu_in", 0.0), unit),
                        "mu_out_fmt": _fmt_kpi(stats.get("mu_out", 0.0), unit),

                        "definition": _behavior_to_text(p.pattern),
                    }

                    if include_pattern_graphs:
                        pattern_row["graph"] = _behavior_to_cy(p.pattern)

                    pattern_rows.append(pattern_row)

                pattern_rows.sort(key=lambda r: abs(r["delta"]), reverse=True)

                row["patterns"] = pattern_rows

            rows.append(row)

        return jsonify({
            "status": "ok",
            "kpi": kpi,
            "kpi_unit": kpi_unit_of.get(kpi, ""),
            "only_interesting": only_interesting,
            "include_leaves": include_leaves,
            "include_behaviors": include_patterns,
            "include_behavior_graphs": include_pattern_graphs,
            "n_cfgs": len(rows),
            "rows": rows,
        })

    @app.get("/api/tree/cfg/<int:ci>/leaf/<int:leaf_id>")
    def api_tree_leaf(ci: int, leaf_id: int):
        if not (0 <= ci < len(state.bundle.cfg_runs)):
            abort(404)

        b = state.bundle

        kpi = request.args.get("kpi", b.primary_kpi)
        if kpi not in b.kpi_values:
            kpi = b.primary_kpi

        cache = getattr(state, "_leaves_cache", None)
        if cache is None:
            cache = {}
            state._leaves_cache = cache

        key = (ci, kpi)
        payload = cache.get(key)

        if payload is None:
            payload = compute_leaf_membership(state, ci, kpi=kpi)
            cache[key] = payload

        for leaf in payload.get("leaves", []):
            if int(leaf.get("leaf_id")) == leaf_id:
                return jsonify({
                    "status": "ok",
                    "cfg_idx": ci,
                    "cfg": list(b.cfg_runs[ci].cfg),
                    "cfg_str": fmt_cfg(b.types, b.level_names, b.cfg_runs[ci].cfg),
                    "K": b.cfg_runs[ci].K,
                    "kpi": kpi,
                    "kpi_unit": kpi_unit_of.get(kpi, ""),
                    "leaf": leaf,
                })

        return jsonify({
            "status": "not_found",
            "cfg_idx": ci,
            "leaf_id": leaf_id,
            "available_leaf_ids": [
                leaf.get("leaf_id") for leaf in payload.get("leaves", [])
            ],
        }), 404

    @app.get("/api/behavior/<int:ci>/<int:pi>")
    def api_pattern(ci: int, pi: int):
        if not (0 <= ci < len(state.bundle.cfg_runs)):
            abort(404)
        run = state.bundle.cfg_runs[ci]
        if not (0 <= pi < len(run.patterns)):
            abort(404)
        b = state.bundle
        p = run.patterns[pi]
        kpi = request.args.get("kpi", b.primary_kpi)
        if kpi not in b.kpi_values:
            kpi = b.primary_kpi
        unit = kpi_unit_of.get(kpi, "")
        kpi_vals = b.kpi_values.get(kpi, [])

        in_vals  = [kpi_vals[i] for i in p.in_exec_idx if i < len(kpi_vals)]
        out_vals = [kpi_vals[i] for i in p.out_exec_idx if i < len(kpi_vals)]
        return jsonify({
            "cfg_idx": ci,
            "behavior_idx": pi,
            "cfg_str": fmt_cfg(b.types, b.level_names, run.cfg),
            "size_nodes": p.size_nodes,
            "size_edges": p.size_edges,
            "support": p.support,
            "n_in":  len(p.in_exec_idx),
            "n_out": len(p.out_exec_idx),
            "in_exec_idx":  list(p.in_exec_idx),
            "out_exec_idx": list(p.out_exec_idx),
            "kpi": kpi,
            "kpi_unit": unit,
            "kpi_in":  _summary_stats(in_vals),
            "kpi_out": _summary_stats(out_vals),
            "kpi_stats_raw": p.kpi_stats.get(kpi, {}),
            "graph": _behavior_to_cy(p.pattern),
        })

    @app.get("/api/behavior/<int:ci>/<int:pi>/attrs")
    def api_pattern_attrs(ci: int, pi: int):
        if not (0 <= ci < len(state.bundle.cfg_runs)):
            abort(404)
        run = state.bundle.cfg_runs[ci]
        if not (0 <= pi < len(run.patterns)):
            abort(404)
        kpi = request.args.get("kpi", state.bundle.primary_kpi)
        return jsonify(attribute_breakdown(state, ci, pi, kpi=kpi))

    @app.get("/api/behavior/<int:ci>/<int:pi>/debug")
    def api_pattern_debug(ci: int, pi: int):
        if not (0 <= ci < len(state.bundle.cfg_runs)):
            abort(404)
        run = state.bundle.cfg_runs[ci]
        if not (0 <= pi < len(run.patterns)):
            abort(404)
        b = state.bundle
        p = run.patterns[pi]
        nodes = []
        for n, d in p.pattern.nodes(data=True):
            nodes.append({"id": n,
                          "data": {k: (list(v) if isinstance(v, frozenset)
                                       else v) for k, v in d.items()}})
        edges = []
        for u, v, d in p.pattern.edges(data=True):
            edges.append({
                "source": u, "target": v,
                "data": {k: (sorted(list(v_)) if isinstance(v_, frozenset)
                              else v_) for k, v_ in d.items()},
            })
        return jsonify({
            "cfg_idx": ci,
            "behavior_idx": pi,
            "cfg":     list(run.cfg),
            "cfg_str": fmt_cfg(b.types, b.level_names, run.cfg),
            "support":     int(p.support),
            "size_nodes":  int(p.size_nodes),
            "size_edges":  int(p.size_edges),
            "n_in":        len(p.in_exec_idx),
            "n_out":       len(p.out_exec_idx),
            "in_exec_idx":  list(p.in_exec_idx),
            "out_exec_idx": list(p.out_exec_idx),
            "n_executions": len(b.executions),
            "signature": list(p.signature) if p.signature is not None else None,
            "kpi_stats": p.kpi_stats,
            "graph_nodes": nodes,
            "graph_edges": edges,
        })

    @app.get("/api/behavior/pair/<int:ci>/<int:pa>/<int:pb>/debug")
    def api_pattern_pair_debug(ci: int, pa: int, pb: int):
        if not (0 <= ci < len(state.bundle.cfg_runs)):
            abort(404)
        run = state.bundle.cfg_runs[ci]
        if not (0 <= pa < len(run.patterns) and 0 <= pb < len(run.patterns)):
            abort(404)
        A = set(run.patterns[pa].in_exec_idx)
        B = set(run.patterns[pb].in_exec_idx)
        U = set(range(len(state.bundle.executions)))
        return jsonify({
            "cfg_idx": ci, "pa": pa, "pb": pb,
            "n_executions": len(U),
            "n_a": len(A), "n_b": len(B),
            "n_a_intersect_b": len(A & B),
            "n_a_union_b": len(A | B),
            "n_a_only": len(A - B),
            "n_b_only": len(B - A),
            "n_neither": len(U - (A | B)),
            "is_disjoint": len(A & B) == 0,
            "covers_all":   (A | B) == U,
            "is_perfect_complement": (A == U - B) and bool(A) and bool(B),
        })

    @app.get("/api/lattice/options")
    def api_lattice_options():
        b = state.bundle
        useful: List[List[int]] = []
        for ci, run in enumerate(b.cfg_runs):
            if not run.is_interesting:
                continue
            kept = state.behaviors_after_subsumption(ci)
            n_kept = (len(kept) if kept is not None else len(run.patterns))
            if n_kept < 1:
                continue
            useful.append({
                "cfg_idx": ci,
                "cfg": list(run.cfg),
                "cfg_str": fmt_cfg(b.types, b.level_names, run.cfg),
                "K": run.K,
                "n_abstractions": n_kept,
            })
        return jsonify({
            "types": b.types,
            "level_names": b.level_names,
            "useful_cfgs": useful,
            "n_useful": len(useful),
        })

    @app.get("/api/tree/summary")
    def api_tree_summary():
        b = state.bundle
        kpi = request.args.get("kpi", b.primary_kpi)
        if kpi not in b.kpi_values:
            kpi = b.primary_kpi
        force = bool(request.args.get("force", type=int, default=0))

        cache = getattr(state, "_tree_summary_cache", {})
        if not force and kpi in cache:
            return jsonify(cache[kpi])

        rows = analyze_all_cfgs(state, kpi=kpi)
        payload = {
            "kpi": kpi,
            "kpi_unit": kpi_unit_of.get(kpi, ""),
            "n_rows": len(rows),
            "rows": rows,
        }
        cache[kpi] = payload
        state._tree_summary_cache = cache
        return jsonify(payload)

    @app.get("/api/tree/cfg/<int:ci>")
    def api_tree_cfg(ci: int):
        if not (0 <= ci < len(state.bundle.cfg_runs)):
            abort(404)
        kpi = request.args.get("kpi", state.bundle.primary_kpi)
        if kpi not in state.bundle.kpi_values:
            kpi = state.bundle.primary_kpi
        return jsonify(summarize_cfg_tree(state, ci, kpi=kpi))

    @app.get("/api/tree/cfg/<int:ci>/relation")
    def api_tree_relation(ci: int):
        if not (0 <= ci < len(state.bundle.cfg_runs)):
            abort(404)
        try:
            p1 = int(request.args.get("p1", "-1"))
            p2 = int(request.args.get("p2", "-1"))
        except ValueError:
            return jsonify({"status": "bad_params"}), 400
        return jsonify(tree_behavior_relation(state, ci, p1, p2))

    @app.get("/api/tree/cfg/<int:ci>/leaves")
    def api_tree_leaves(ci: int):
        if not (0 <= ci < len(state.bundle.cfg_runs)):
            abort(404)
        kpi = request.args.get("kpi", state.bundle.primary_kpi)
        if kpi not in state.bundle.kpi_values:
            kpi = state.bundle.primary_kpi
        cache = getattr(state, "_leaves_cache", None)
        if cache is None:
            cache = {}
            state._leaves_cache = cache
        key = (ci, kpi)
        payload = cache.get(key)
        if payload is None:
            payload = compute_leaf_membership(state, ci, kpi=kpi)
            cache[key] = payload
        return jsonify(payload)

    @app.get("/api/tree/index/flat")
    def api_tree_index_flat():
        b = state.bundle

        kpi = request.args.get("kpi", b.primary_kpi)
        if kpi not in b.kpi_values:
            kpi = b.primary_kpi

        cache = getattr(state, "_leaves_cache", None)
        if cache is None:
            cache = {}
            state._leaves_cache = cache

        def _pattern_condition_from_step(step: Dict[str, Any]) -> str:
            pid = (
                step.get("behavior_idx")
                if "behavior_idx" in step
                else step.get("pattern_id")
                if "pattern_id" in step
                else step.get("pi")
            )

            positive = None

            if "present" in step:
                positive = bool(step["present"])
            elif "is_present" in step:
                positive = bool(step["is_present"])
            elif "value" in step:
                positive = bool(step["value"])
            elif "branch" in step:
                br = str(step["branch"]).lower()
                if br in {"true", "yes", "present", "in", "right", "1"}:
                    positive = True
                elif br in {"false", "no", "absent", "out", "left", "0"}:
                    positive = False

            if pid is not None:
                atom = f"p{pid}"
                return f"NOT {atom}" if positive is False else atom

            return str(step.get("human_condition", step))

        def _leaf_boolean_label(leaf: Dict[str, Any]) -> str:
            steps = leaf.get("path_behaviors", []) or []
            parts = [_pattern_condition_from_step(s) for s in steps]
            parts = [p for p in parts if p and p != "{}"]
            return " AND ".join(parts) if parts else "ROOT"

        rows = []

        for ci, run in enumerate(b.cfg_runs):
            if not run.is_interesting:
                continue

            kept = state.behaviors_after_subsumption(ci)
            n_kept = len(kept) if kept is not None else len(run.patterns)
            if n_kept < 1:
                continue

            key = (ci, kpi)
            payload = cache.get(key)
            if payload is None:
                payload = compute_leaf_membership(state, ci, kpi=kpi)
                cache[key] = payload

            for leaf in payload.get("leaves", []):
                exec_idx = list(leaf.get("exec_idx", []))
                rows.append({
                    "cfg_idx": ci,
                    "cfg": list(run.cfg),
                    "cfg_str": fmt_cfg(b.types, b.level_names, run.cfg),
                    "K": run.K,
                    "leaf_id": leaf.get("leaf_id"),
                    "leaf_name": f"L{leaf.get('leaf_id')}",
                    "leaf": _leaf_boolean_label(leaf),
                    "human_label": leaf.get(
                        "human_label",
                        f"L{leaf.get('leaf_id')}",
                    ),
                    "n_exec": len(exec_idx),
                    "exec_idx": exec_idx,
                })

        return jsonify({
            "kpi": kpi,
            "kpi_unit": kpi_unit_of.get(kpi, ""),
            "n_rows": len(rows),
            "rows": rows,
        })

    @app.get("/api/tree/cfg/<int:ci>/leaf/<int:leaf_id>/cross")
    def api_tree_cross(ci: int, leaf_id: int):
        if not (0 <= ci < len(state.bundle.cfg_runs)):
            abort(404)

        kpi = request.args.get("kpi", state.bundle.primary_kpi)
        if kpi not in state.bundle.kpi_values:
            kpi = state.bundle.primary_kpi

        try:
            top_n_arg = request.args.get("top_n", type=int)
        except ValueError:
            top_n_arg = None

        target_cfg = request.args.get("target_cfg", type=int)

        if target_cfg is not None:
            if not (0 <= target_cfg < len(state.bundle.cfg_runs)):
                return jsonify({
                    "status": "bad_target_cfg",
                    "target_cfg": target_cfg,
                    "n_cfg_runs": len(state.bundle.cfg_runs),
                }), 400

        cache = getattr(state, "_leaves_cache", None)
        if cache is None:
            cache = {}
            state._leaves_cache = cache

        payload = cross_leaf_distribution(
            state,
            source_cfg_idx=ci,
            source_leaf_id=leaf_id,
            kpi=kpi,
            top_n=top_n_arg,
            cache=cache,
        )

        if target_cfg is not None:
            targets = payload.get("targets", [])
            payload["targets"] = [
                t for t in targets
                if t.get("cfg_idx") == target_cfg
            ]
            payload["target_cfg"] = target_cfg
            payload["n_targets"] = len(payload["targets"])

        return jsonify(payload)

    @app.get("/api/tree/cfg/<int:ci>/debug/full-cross")
    def api_tree_cfg_debug_full_cross(ci: int):
        if not (0 <= ci < len(state.bundle.cfg_runs)):
            abort(404)

        b = state.bundle

        kpi = request.args.get("kpi", b.primary_kpi)
        if kpi not in b.kpi_values:
            kpi = b.primary_kpi

        try:
            top_n_arg = request.args.get("top_n", type=int)
        except ValueError:
            top_n_arg = None

        cache = getattr(state, "_leaves_cache", None)
        if cache is None:
            cache = {}
            state._leaves_cache = cache

        key = (ci, kpi)
        source_payload = cache.get(key)
        if source_payload is None:
            source_payload = compute_leaf_membership(state, ci, kpi=kpi)
            cache[key] = source_payload

        print("\n" + "=" * 80)
        print(f"DEBUG FULL CROSS — cfg {ci}")
        print(f"Configuration: {fmt_cfg(b.types, b.level_names, b.cfg_runs[ci].cfg)}")
        print(f"KPI: {kpi}")
        print("=" * 80)

        all_results = []

        for leaf in source_payload.get("leaves", []):
            leaf_id = leaf["leaf_id"]

            print("\n" + "-" * 80)
            print(f"FOGLIA SORGENTE {leaf_id}")
            print("-" * 80)

            print("Executions in leaf:")
            print(leaf.get("exec_idx", []))

            print("\nCaratterizzazione foglia:")
            print(leaf.get("human_label", f"L{leaf_id}"))

            print("\nPattern / condizioni che portano alla foglia:")
            for step in leaf.get("path_behaviors", []):
                print(" -", step.get("human_condition", step))

            cross = cross_leaf_distribution(
                state,
                source_cfg_idx=ci,
                source_leaf_id=leaf_id,
                kpi=kpi,
                top_n=top_n_arg,
                cache=cache,
            )

            print("\nWhere these executions end up in other trees:")

            for target in cross.get("targets", []):
                target_ci = target.get("cfg_idx")
                target_cfg = b.cfg_runs[target_ci].cfg

                print("\nTarget configuration:", target_ci)
                print("rho =", fmt_cfg(b.types, b.level_names, target_cfg))

                for target_leaf in target.get("leaves", []):
                    print("  Foglia:", target_leaf.get("leaf_id"))
                    print("  Executions:", target_leaf.get("exec_idx", []))
                    print("  Pattern della foglia:")

                    for step in target_leaf.get("path_behaviors", []):
                        print("   -", step.get("human_condition", step))

            all_results.append({
                "source_leaf": leaf,
                "cross": cross,
            })

        return jsonify({
            "status": "ok",
            "cfg_idx": ci,
            "cfg": list(b.cfg_runs[ci].cfg),
            "cfg_str": fmt_cfg(b.types, b.level_names, b.cfg_runs[ci].cfg),
            "kpi": kpi,
            "n_source_leaves": len(source_payload.get("leaves", [])),
            "results": all_results,
        })

    return app

def main(argv: List[str] = None) -> int:
    ap = argparse.ArgumentParser(
        description="3-level OCEL explorer "
                    "(Lattice -> Pattern -> Attributes)",
    )
    ap.add_argument("bundle", help="Path to a run bundle .pkl file")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8050)
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args(argv)

    print(f"loading bundle: {args.bundle} …", flush=True)

    import __main__

    try:
        import rho_lift
    except ImportError:
        import rho_lift_gspan_only as rho_lift

    for name in [
        "RunBundle",
        "BundleCfgRun",
        "BundlePattern",
        "LatticeRow",
        "Hierarchy",
        "KpiSpec",
    ]:
        if hasattr(rho_lift, name):
            setattr(__main__, name, getattr(rho_lift, name))

    state = ExplorerState(args.bundle)

    print_rho_dependencies(state)

    b = state.bundle
    print(f"{len(b.executions)} executions, {getattr(b, 'n_behaviors_total', getattr(b, 'n_behaviors_total', 0))} "
          f"patterns across {b.n_cfg_total} configs "
          f"({b.n_cfg_interesting} interesting)")
    print(f"  primary KPI: {b.primary_kpi}")

    app = create_app(state)
    print(f"\nopen http://{args.host}:{args.port}/ in your browser")
    app.run(host=args.host, port=args.port, debug=args.debug)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
