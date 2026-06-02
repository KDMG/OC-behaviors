#!/usr/bin/env python3

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from statistics import mean
from typing import Any, Callable, Dict, List, Optional, Tuple


@dataclass
class KPI:
    name: str
    unit: str
    func: Callable[[Any], float]
    description: str = ""

def _safe_timestamps(ex) -> "Optional[Any]":
    import pandas as pd
    if not ex.event_info:
        return None
    ts_values = [info.get("timestamp") for info in ex.event_info.values()]
    if not ts_values:
        return None
    try:
        s = pd.to_datetime(pd.Series(ts_values), utc=True, errors="coerce").dropna()
    except Exception:
        return None
    return s if not s.empty else None


def kpi_duration(ex) -> float:
    ts = _safe_timestamps(ex)
    if ts is None:
        return 0.0
    return float((ts.max() - ts.min()).total_seconds())

def kpi_n_events(ex) -> float:
    return float(len(ex.events))

def kpi_n_objects(ex) -> float:
    return float(len(ex.objects))

def kpi_n_activities(ex) -> float:
    acts = {info.get("activity") for info in ex.event_info.values()
            if info.get("activity") is not None}
    return float(len(acts))

def kpi_event_density(ex) -> float:
    d = kpi_duration(ex)
    if d <= 0:
        return 0.0
    return kpi_n_events(ex) / d

BUILTIN_KPIS: Dict[str, KPI] = {
    "duration": KPI("duration", "seconds", kpi_duration, "Duration (max_ts − min_ts)."),
    "n_events": KPI("n_events", "events", kpi_n_events, "Number of events."),
    "n_objects": KPI("n_objects", "objects", kpi_n_objects, "Number of objects."),
    "n_activities": KPI("n_activities", "acts", kpi_n_activities, "Number of activities."),
    "event_density": KPI("event_density", "ev/s", kpi_event_density, "Events per second."),
}

_CUSTOM_RE = re.compile(r"^(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s*:=\s*(?P<expr>.+)$", re.S)


def _make_custom_kpi(name: str, expr: str) -> KPI:
    code = compile(expr, f"<kpi:{name}>", "eval")

    def _call(ex):
        import pandas as pd
        ts = _safe_timestamps(ex)
        if ts is not None and not ts.empty:
            ts_min = ts.min()
            ts_max = ts.max()
            duration = float((ts_max - ts_min).total_seconds())
        else:
            ts_min = ts_max = None
            duration = 0.0
        acts = {info.get("activity") for info in ex.event_info.values()
                if info.get("activity") is not None}
        scope = {
            "ex": ex,
            "events": list(ex.events),
            "objects": list(ex.objects),
            "edges": list(ex.edges),
            "duration": duration,
            "n_events": len(ex.events),
            "n_objects": len(ex.objects),
            "activities": list(acts),
            "ts_min": ts_min,
            "ts_max": ts_max,
            "pd": pd,
            "math": math,
        }
        try:
            v = eval(code, {"__builtins__": {}}, scope)
        except Exception as exn:
            raise RuntimeError(
                f"Error evaluating KPI {name!r}: {exn!s}"
            ) from exn
        try:
            return float(v)
        except (TypeError, ValueError):
            return 0.0

    return KPI(name=name, unit="custom", func=_call,
               description=f"custom: {expr}")

def parse_kpi_flag(spec: str, default_counter: List[int]) -> KPI:
    spec = spec.strip()

    if spec in BUILTIN_KPIS:
        return BUILTIN_KPIS[spec]

    m = _CUSTOM_RE.match(spec)
    if m:
        return _make_custom_kpi(m.group("name"), m.group("expr"))

    default_counter[0] += 1
    return _make_custom_kpi(f"kpi_{default_counter[0]}", spec)

def resolve_kpi_flags(specs: List[str]) -> List[KPI]:
    if not specs:
        return [BUILTIN_KPIS["duration"]]
    counter = [0]
    resolved = [parse_kpi_flag(s, counter) for s in specs]
    seen = set()
    out: List[KPI] = []
    for k in resolved:
        if k.name in seen:
            continue
        seen.add(k.name)
        out.append(k)
    return out

def compute_kpi_matrix(
    executions:   List[Any],
    kpis:         List[KPI],
) -> Dict[str, List[float]]:
    return {
        k.name: [k.func(ex) for ex in executions]
        for k in kpis
    }


def summarize_kpi(values: List[float]) -> Dict[str, float]:
    if not values:
        return {"n": 0, "min": 0.0, "max": 0.0, "mean": 0.0, "median": 0.0}
    from statistics import median
    return {
        "n": len(values),
        "min": min(values),
        "max": max(values),
        "mean": mean(values),
        "median": median(values),
    }
