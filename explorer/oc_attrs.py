from __future__ import annotations

import math
from collections import Counter, defaultdict
from statistics import median
from typing import Any, Dict, List, Optional

def _is_nan(value: Any) -> bool:
    return isinstance(value, float) and math.isnan(value)


def _mode(values: List[Any]) -> Any:
    values = [value for value in values if value is not None and not _is_nan(value)]
    if not values:
        return None
    return Counter(str(value) for value in values).most_common(1)[0][0]


def execution_attribute_fingerprint(
    execution,
    bundle,
    event_attributes_by_id: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    fingerprint: Dict[str, Any] = {}

    objects_by_type: Dict[str, List[str]] = defaultdict(list)
    for object_id in execution.objects:
        object_type = bundle.obj_types_map.get(object_id)
        if object_type is not None:
            objects_by_type[object_type].append(object_id)

    for object_type, attributes in bundle.per_type_attrs.items():
        object_ids = objects_by_type.get(object_type, [])
        for attribute_name, values_by_object_id in attributes:
            values = [
                values_by_object_id.get(object_id)
                for object_id in object_ids
                if object_id in values_by_object_id
            ]
            fingerprint[f"{object_type}.{attribute_name}"] = _mode(values)

    if event_attributes_by_id:
        values_by_column: Dict[str, List[Any]] = defaultdict(list)
        for event_id in execution.events:
            row = event_attributes_by_id.get(str(event_id))
            if not row:
                continue
            for column, value in row.items():
                values_by_column[column].append(value)

        for column, values in values_by_column.items():
            fingerprint[column] = _mode(values)

    return fingerprint


def _value_distribution(
    indexes: List[int],
    fingerprints: List[Dict[str, Any]],
    attribute: str,
) -> Counter:
    return Counter(
        str(fingerprints[index].get(attribute))
        for index in indexes
        if fingerprints[index].get(attribute) is not None
    )


def _rank_attribute_shifts(
    group_a_indexes: List[int],
    group_b_indexes: List[int],
    fingerprints: List[Dict[str, Any]],
    attributes: List[str],
    group_a_label: str,
    group_b_label: str,
    top_values: int = 6,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    n_a = len(group_a_indexes)
    n_b = len(group_b_indexes)

    for attribute in attributes:
        dist_a = _value_distribution(group_a_indexes, fingerprints, attribute)
        dist_b = _value_distribution(group_b_indexes, fingerprints, attribute)
        if not dist_a and not dist_b:
            continue

        value_rows: List[Dict[str, Any]] = []
        for value in set(dist_a) | set(dist_b):
            freq_a = dist_a[value] / n_a if n_a else 0.0
            freq_b = dist_b[value] / n_b if n_b else 0.0
            value_rows.append({
                "value": value,
                "freq_a": round(freq_a, 4),
                "freq_b": round(freq_b, 4),
                "delta": round(freq_a - freq_b, 4),
                "count_a": dist_a[value],
                "count_b": dist_b[value],
            })

        value_rows.sort(key=lambda row: abs(row["delta"]), reverse=True)
        score = max((abs(row["delta"]) for row in value_rows), default=0.0)
        rows.append({
            "attr": attribute,
            "n_a": n_a,
            "n_b": n_b,
            "label_a": group_a_label,
            "label_b": group_b_label,
            "score": round(score, 4),
            "values": value_rows[:top_values],
            "n_values": len(value_rows),
        })

    rows.sort(key=lambda row: row["score"], reverse=True)
    return rows


def _available_attributes(fingerprints: List[Dict[str, Any]]) -> List[str]:
    seen: Dict[str, int] = {}
    for fingerprint in fingerprints:
        for attribute, value in fingerprint.items():
            if value is not None:
                seen[attribute] = seen.get(attribute, 0) + 1
    return sorted(attribute for attribute, count in seen.items() if count >= 1)


def _split_by_kpi_median(
    indexes: List[int],
    kpi_values: List[float],
) -> tuple[List[int], List[int]]:
    values = [kpi_values[index] for index in indexes]
    split_value = median(values)
    high = []
    low = []

    for index, value in zip(indexes, values):
        if value > split_value:
            high.append(index)
        elif value < split_value:
            low.append(index)

    return high, low


def attribute_breakdown(
    state,
    cfg_idx: int,
    pattern_idx: int,
    kpi: Optional[str] = None,
) -> Dict[str, Any]:
    bundle = state.bundle
    pattern = bundle.cfg_runs[cfg_idx].patterns[pattern_idx]

    if kpi is None or kpi not in bundle.kpi_values:
        kpi = bundle.primary_kpi

    _, event_attributes_by_id = state.ensure_events_df()
    event_attributes_by_id = event_attributes_by_id or {}

    n_executions = len(bundle.executions)
    fingerprints = [
        execution_attribute_fingerprint(execution, bundle, event_attributes_by_id)
        for execution in bundle.executions
    ]
    attributes = _available_attributes(fingerprints)

    in_indexes = list(pattern.in_exec_idx)
    out_indexes = list(pattern.out_exec_idx)
    if not out_indexes:
        in_set = set(in_indexes)
        out_indexes = [index for index in range(n_executions) if index not in in_set]

    if in_indexes and out_indexes:
        in_vs_out = _rank_attribute_shifts(
            in_indexes,
            out_indexes,
            fingerprints,
            attributes,
            group_a_label="contains pattern",
            group_b_label="does NOT contain pattern",
        )
        note_in_vs_out = ""
    else:
        in_vs_out = []
        note_in_vs_out = (
            "no executions contain the pattern"
            if not in_indexes
            else "every execution contains the pattern (no OUT group)"
        )

    high_indexes: List[int] = []
    low_indexes: List[int] = []
    note_high_vs_low = ""
    kpi_values = bundle.kpi_values.get(kpi, [])

    if len(in_indexes) >= 4 and len(kpi_values) == n_executions:
        high_indexes, low_indexes = _split_by_kpi_median(in_indexes, kpi_values)
        if high_indexes and low_indexes:
            high_vs_low = _rank_attribute_shifts(
                high_indexes,
                low_indexes,
                fingerprints,
                attributes,
                group_a_label=f"high {kpi}",
                group_b_label=f"low {kpi}",
            )
        else:
            high_vs_low = []
            note_high_vs_low = f"all IN executions tie on '{kpi}' median; can't split"
    else:
        high_vs_low = []
        if len(in_indexes) < 4:
            note_high_vs_low = (
                f"only {len(in_indexes)} executions contain the pattern; "
                "need at least 4 to split on the KPI median"
            )
        elif len(kpi_values) != n_executions:
            note_high_vs_low = f"KPI '{kpi}' has no values for this bundle"

    return {
        "cfg_idx": cfg_idx,
        "pattern_idx": pattern_idx,
        "kpi": kpi,
        "n_executions_total": n_executions,
        "n_in": len(in_indexes),
        "n_out": len(out_indexes),
        "n_high": len(high_indexes),
        "n_low": len(low_indexes),
        "in_vs_out": in_vs_out,
        "high_vs_low": high_vs_low,
        "note_in_vs_out": note_in_vs_out,
        "note_high_vs_low": note_high_vs_low,
    }
