from __future__ import annotations

import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _ROOT not in _sys.path:
    _sys.path.insert(0, _ROOT)

from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

from mining.core import LevelConfiguration, OCELLevelGraph, compute_behavior

try:
    from rho_lift import BundlePattern, RunBundle, load_bundle, rebuild_hierarchies_from_attrs
except ModuleNotFoundError:
    from rho_lift_gspan_only import BundlePattern, RunBundle, load_bundle, rebuild_hierarchies_from_attrs


def pretty_level(name: str) -> str:
    return name


def fmt_cfg(types: List[str], level_names: Dict[str, List[str]], cfg: Tuple[int, ...]) -> str:
    parts = []
    for tau, level in zip(types, cfg):
        names = level_names.get(tau, [])
        label = pretty_level(names[level]) if 0 <= level < len(names) else f"lv{level}"
        parts.append(f"{tau}:{label}")
    return ", ".join(parts)


@dataclass
class CfgRow:
    cfg_idx: int
    cfg: List[int]
    cfg_str: str
    K: int
    n_abstractions: int
    n_executions: int
    is_interesting: bool
    best_delta: float


class ExplorerState:
    def __init__(self, bundle_path: str) -> None:
        self.bundle_path = bundle_path
        self.bundle: RunBundle = load_bundle(bundle_path)
        self.hierarchies, _ = rebuild_hierarchies_from_attrs(
            self.bundle.per_type_attrs,
            self.bundle.obj_types_map,
        )
        self.bundle.hierarchies = self.hierarchies
        self.cfg_index: Dict[Tuple[int, ...], int] = {
            tuple(run.cfg): idx for idx, run in enumerate(self.bundle.cfg_runs)
        }
        self.lattice: List[CfgRow] = self._build_lattice_rows()
        self._events_df = None
        self._eid_to_attrs: Optional[Dict[str, Dict[str, Any]]] = None
        self._subsumption_kept: Dict[int, Optional[set[int]]] = {}

    def _build_lattice_rows(self) -> List[CfgRow]:
        bundle = self.bundle
        rows = []
        for cfg_idx, run in enumerate(bundle.cfg_runs):
            best_delta = max(
                (
                    pattern.kpi_stats.get(bundle.primary_kpi, {}).get("delta", 0.0)
                    for pattern in run.patterns
                ),
                default=0.0,
            )
            rows.append(
                CfgRow(
                    cfg_idx=cfg_idx,
                    cfg=list(run.cfg),
                    cfg_str=fmt_cfg(bundle.types, bundle.level_names, run.cfg),
                    K=run.K,
                    n_abstractions=len(run.patterns),
                    n_executions=run.n_executions,
                    is_interesting=run.is_interesting,
                    best_delta=best_delta,
                )
            )
        return rows

    @lru_cache(maxsize=2048)
    def abstraction_for(self, cfg_idx: int, exec_idx: int):
        run = self.bundle.cfg_runs[cfg_idx]
        cfg = tuple(run.cfg)
        level_config = LevelConfiguration(
            dict(zip(self.bundle.types, cfg)),
            self.hierarchies,
            self.bundle.obj_types_map,
        )
        execution = self.bundle.executions[exec_idx]
        return compute_behavior(OCELLevelGraph(execution, level_config))

    def ensure_events_df(self):
        if self._events_df is not None:
            return self._events_df, self._eid_to_attrs

        try:
            import pm4py
        except ImportError:
            self._events_df = None
            self._eid_to_attrs = {}
            return None, {}

        path = self.bundle.ocel_path
        try:
            if path.endswith((".sqlite", ".db", ".sqlite3")):
                if hasattr(pm4py, "read_ocel2_sqlite"):
                    ocel = pm4py.read_ocel2_sqlite(path)
                else:
                    ocel = pm4py.read_ocel2(path)
            else:
                ocel = pm4py.read_ocel(path)
            df = ocel.events.copy()
        except Exception:
            self._events_df = None
            self._eid_to_attrs = {}
            return None, {}

        skip = {
            "ocel:eid",
            "ocel:activity",
            "ocel:timestamp",
            "ocel:type",
            "@@type",
            "@@index",
        }
        attr_cols = [col for col in df.columns if col not in skip]

        eid_to_attrs: Dict[str, Dict[str, Any]] = {}
        for _, row in df.iterrows():
            event_id = str(row["ocel:eid"]) if "ocel:eid" in df.columns else str(row.name)
            eid_to_attrs[event_id] = {col: row[col] for col in attr_cols}

        self._events_df = df
        self._eid_to_attrs = eid_to_attrs
        return df, eid_to_attrs

    def kept_behaviors_for_cfg(self, cfg_idx: int) -> Optional[set[int]]:
        if cfg_idx in self._subsumption_kept:
            return self._subsumption_kept[cfg_idx]

        from explorer.oc_subsume import compute_kept_patterns

        run = self.bundle.cfg_runs[cfg_idx]
        kept = compute_kept_patterns(run.patterns)
        self._subsumption_kept[cfg_idx] = kept
        return kept

    def behaviors_after_subsumption(self, cfg_idx: int) -> List[Tuple[int, BundlePattern]]:
        run = self.bundle.cfg_runs[cfg_idx]
        kept = self.kept_behaviors_for_cfg(cfg_idx)
        if kept is None:
            return list(enumerate(run.patterns))
        return [(idx, beh) for idx, beh in enumerate(run.patterns) if idx in kept]
