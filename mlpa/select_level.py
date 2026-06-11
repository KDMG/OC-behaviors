from __future__ import annotations

import ast
from collections import defaultdict
from typing import Iterable, Optional, Any

import pandas as pd
import networkx as nx
import numpy as np


import sqlite3
from collections import defaultdict

def _load_o2o_neighbors_from_sqlite(
    sqlite_path: str,
    include_same_type: bool = False,
):
    con = sqlite3.connect(sqlite_path)
    cur = con.cursor()

    obj_rows = cur.execute("""
        SELECT ocel_id, ocel_type
        FROM object
    """).fetchall()

    object_to_type = {}
    type_to_objects = defaultdict(set)

    for oid, otype in obj_rows:
        oid = str(oid)
        otype = str(otype)
        object_to_type[oid] = otype
        type_to_objects[otype].add(oid)

    o2o_rows = cur.execute("""
        SELECT ocel_source_id, ocel_target_id
        FROM object_object
    """).fetchall()

    con.close()

    neighbors = defaultdict(lambda: defaultdict(set))

    for src, tgt in o2o_rows:
        src = str(src)
        tgt = str(tgt)

        if src not in object_to_type or tgt not in object_to_type:
            continue

        src_type = object_to_type[src]
        tgt_type = object_to_type[tgt]

        if (not include_same_type) and src_type == tgt_type:
            continue

        neighbors[(src_type, tgt_type)][src].add(tgt)
        neighbors[(tgt_type, src_type)][tgt].add(src)

    return neighbors, {t: sorted(v) for t, v in type_to_objects.items()}

def _infer_direct_mapping_from_o2o(
    neighbors,
    source_type: str,
    target_type: str,
    source_object_ids: list[str],
    mode: str = "strict",
):
    mapping = {}
    missing = []
    ambiguous = {}

    src_to_tgts = neighbors.get((source_type, target_type), {})

    for s in source_object_ids:
        tgts = src_to_tgts.get(s, {'no_'+target_type.lower()})

        if not tgts:
            missing.append(s)
            continue

        if len(tgts) == 1:
            mapping[s] = next(iter(tgts))
        else:
            if mode == "most_frequent":
                mapping[s] = sorted(tgts)[0]
            else:
                ambiguous[s] = sorted(tgts)

    if missing:
        raise ValueError(
            f"Impossibile ricavare il mapping {source_type} -> {target_type} "
            f"for {len(missing)} objects. Examples: {missing[:10]}"
        )

    if ambiguous:
        ex = list(ambiguous.items())[:5]
        raise ValueError(
            f"The mapping {source_type} -> {target_type} is not unique. "
            f"Esempi: {ex}"
        )

    return mapping


def _build_all_direct_mappings_from_sqlite(
    sqlite_path: str,
    type_links: dict[str, list[str]],
    mode: str = "strict",
):
    all_types = sorted(_all_nodes(type_links))
    neighbors, object_ids_by_type = _load_o2o_neighbors_from_sqlite(sqlite_path)

    direct = {}

    for src, tgts in type_links.items():
        if src not in object_ids_by_type:
            raise ValueError(f"Type '{src}' not found in log objects.")

        for tgt in tgts:
            if tgt not in object_ids_by_type:
                raise ValueError(f"Type '{tgt}' not found in log objects.")

            direct[(src, tgt)] = _infer_direct_mapping_from_o2o(
                neighbors=neighbors,
                source_type=src,
                target_type=tgt,
                source_object_ids=object_ids_by_type[src],
                mode=mode,
            )

    return direct, object_ids_by_type

def _pick_col(df: pd.DataFrame, candidates: list[str], required: bool = True) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    if required:
        raise ValueError(
            f"None of the expected columns found. Expected one of: {candidates}. "
            f"Colonne disponibili: {list(df.columns)}"
        )
    return None

def _filter_relations_by_activity_object_rules(
    events_df: pd.DataFrame,
    objects_df: pd.DataFrame,
    relations_df: pd.DataFrame,
    activity_object_rules: dict[str, set[str]] | None,
) -> pd.DataFrame:
    """
    Per ogni activity presente in activity_object_rules,
    mantiene solo i link verso i tipi oggetto consentiti.
    Le activity non presenti nelle regole restano invariate.
    """
    if not activity_object_rules:
        return relations_df.copy()

    # event columns
    ev_id_col = _pick_col(events_df, ["event_id", "ocel:eid", "eid"])
    ev_act_col = _pick_col(events_df, ["event_activity", "ocel:activity", "activity"])

    # object columns
    obj_id_col = _pick_col(objects_df, ["object_id", "ocel:oid", "oid"])
    obj_type_col = _pick_col(objects_df, ["object_type", "ocel:type", "type"])

    # relation columns
    rel_ev_col = _pick_col(relations_df, [ev_id_col, "event_id", "ocel:eid", "eid"])
    rel_obj_col = _pick_col(relations_df, [obj_id_col, "object_id", "ocel:oid", "oid"])

    ev_small = events_df[[ev_id_col, ev_act_col]].drop_duplicates()
    obj_small = objects_df[[obj_id_col, obj_type_col]].drop_duplicates()

    rel = relations_df.merge(
        ev_small,
        left_on=rel_ev_col,
        right_on=ev_id_col,
        how="left",
        validate="many_to_one",
    ).merge(
        obj_small,
        left_on=rel_obj_col,
        right_on=obj_id_col,
        how="left",
        validate="many_to_one",
    )

    mask_keep = rel.apply(
        lambda row: (
            True
            if row[ev_act_col] not in activity_object_rules
            else row[obj_type_col] in activity_object_rules[row[ev_act_col]]
        ),
        axis=1,
    )

    rel = rel[mask_keep].copy()
    return rel[relations_df.columns].copy()

def _safe_isna(x) -> bool:
    """
    Restituisce True solo per scalari nulli/NaN.
    Per strutture multi-valore restituisce sempre False.
    """
    if x is None:
        return True

    if isinstance(x, (list, tuple, set, frozenset, dict, np.ndarray, pd.Series, pd.Index)):
        return False

    try:
        return bool(pd.isna(x))
    except Exception:
        return False


def _norm_cell(x) -> Set[str]:
    """
    Normalizza una cella di una tabella OCEL larga in un insieme di object ids.
    Gestisce:
    - None / NaN
    - liste, tuple, set, ndarray, Series, Index
    - stringhe singole
    - stringhe che rappresentano liste/set/tuple
    """
    if x is None:
        return set()

    if isinstance(x, str):
        s = x.strip()
        if s == "" or s.lower() == "nan":
            return set()

        if s[0] in "[{(" and s[-1] in "]})":
            try:
                parsed = ast.literal_eval(s)
                if isinstance(parsed, (list, tuple, set, frozenset, np.ndarray, pd.Series, pd.Index)):
                    out = set()
                    for v in parsed:
                        if v is None or _safe_isna(v):
                            continue
                        out.add(str(v))
                    return out
            except Exception:
                pass

        return {s}

    if isinstance(x, (list, tuple, set, frozenset, np.ndarray, pd.Series, pd.Index)):
        out = set()
        for v in x:
            if v is None or _safe_isna(v):
                continue
            out.add(str(v))
        return out

    if _safe_isna(x):
        return set()

    return {str(x)}


def extract_process_executions_connected_components_ocpa(
    ocel,
    selected_object_types: Iterable[str],
    selected_activities: Optional[Iterable[str]] = None,
):
    """
    Estrae le process executions come connected components
    su un OCEL OCPA, dopo aver filtrato:
    - i tipi di oggetto
    - activities

    Parametri
    ---------
    ocel : oggetto OCPA OCEL
    selected_object_types : iterable di nomi dei tipi oggetto da considerare
    selected_activities : iterable of activities to consider; if None, keep all

    Ritorna
    -------
    list[dict], dove ogni dict contiene:
      - pe_id
      - events               : lista ordinata di event ids
      - activities           : sorted list of activities
      - objects_by_type      : dict tipo -> lista object ids
      - event_rows           : DataFrame delle righe evento della componente
      - bipartite_graph      : nx.Graph bipartito della componente
    """
    selected_object_types = list(selected_object_types)
    selected_activities = None if selected_activities is None else set(selected_activities)

    df = ocel.log.log.copy()

    # Typical OCPA / OCEL columns
    event_id_col = _pick_col(df, ["event_id", "ocel:eid", "eid"])
    activity_col = _pick_col(df, ["event_activity", "ocel:activity", "activity"])
    timestamp_col = _pick_col(
        df,
        ["event_timestamp", "ocel:timestamp", "timestamp"],
        required=False
    )

    missing_types = [ot for ot in selected_object_types if ot not in df.columns]
    if missing_types:
        raise ValueError(
            f"The following object types are not columns of the OCPA log: {missing_types}\n"
            f"Colonne disponibili: {list(df.columns)}"
        )

    # 1) filter activities
    if selected_activities is not None:
        df = df[df[activity_col].isin(selected_activities)].copy()

    if df.empty:
        return []

    # 2) build bipartite event-object graph
    B = nx.Graph()

    # keep a lightweight version of the df indexed by event_id
    event_rows_dict = {}

    for _, row in df.iterrows():
        eid = str(row[event_id_col])
        act = row[activity_col]
        ts = row[timestamp_col] if timestamp_col is not None else None

        # selected objects incident on the event
        incident_objects = []

        for ot in selected_object_types:
            for oid in _norm_cell(row[ot]):
                incident_objects.append((ot, oid))

        # If the event touches no object of the selected types, skip it
        if not incident_objects:
            continue

        event_node = f"e::{eid}"
        B.add_node(
            event_node,
            kind="event",
            event_id=eid,
            activity=act,
            timestamp=ts,
        )

        event_rows_dict[eid] = row.to_dict()

        for ot, oid in incident_objects:
            obj_node = f"o::{ot}::{oid}"
            B.add_node(
                obj_node,
                kind="object",
                object_id=oid,
                object_type=ot,
            )
            B.add_edge(event_node, obj_node)

    if B.number_of_nodes() == 0:
        return []

    # 3) connected components
    components = []
    for pe_idx, cc_nodes in enumerate(nx.connected_components(B)):
        sub = B.subgraph(cc_nodes).copy()

        events = []
        objects_by_type = defaultdict(set)

        for n, attrs in sub.nodes(data=True):
            if attrs["kind"] == "event":
                events.append(attrs["event_id"])
            else:
                objects_by_type[attrs["object_type"]].add(attrs["object_id"])

        # rebuild the event dataframe for the component
        event_rows = pd.DataFrame([event_rows_dict[eid] for eid in events])

        if timestamp_col is not None and timestamp_col in event_rows.columns:
            event_rows = event_rows.sort_values([timestamp_col, event_id_col], kind="stable")
        else:
            event_rows = event_rows.sort_values([event_id_col], kind="stable")

        ordered_events = event_rows[event_id_col].astype(str).tolist()
        ordered_activities = event_rows[activity_col].tolist()

        components.append(
            {
                "pe_id": pe_idx,
                "events": ordered_events,
                "activities": ordered_activities,
                "objects_by_type": {
                    ot: sorted(list(oids))
                    for ot, oids in objects_by_type.items()
                },
                "event_rows": event_rows.reset_index(drop=True),
                "bipartite_graph": sub,
            }
        )

    # optional: sort PEs by descending number of events
    components.sort(key=lambda x: len(x["events"]), reverse=True)

    # riallineo gli id
    for i, pe in enumerate(components):
        pe["pe_id"] = i

    return components



import ast
import json
from pathlib import Path
from collections import defaultdict, Counter, deque
from typing import Dict, List, Any, Set, Tuple

import pandas as pd


EV_ID = "ocel:eid"
OBJ_ID = "ocel:oid"
OBJ_TYPE = "ocel:type"


# ============================================================
# Accesso robusto all'OCEL
# ============================================================

def _get_log_df(ocel) -> pd.DataFrame:
    """
    Returns a 'wide' event table if available.
    Prova, nell'ordine:
    - ocel.log.log
    - ocel.log
    - ocel.events
    - ocel.get_extended_table()
    """
    candidates = [
        lambda x: getattr(getattr(x, "log", None), "log", None),
        lambda x: getattr(x, "log", None),
        lambda x: getattr(x, "events", None),
    ]

    for getter in candidates:
        try:
            obj = getter(ocel)
            if isinstance(obj, pd.DataFrame):
                return obj.copy()
        except Exception:
            pass

    if hasattr(ocel, "get_extended_table"):
        try:
            df = ocel.get_extended_table()
            if isinstance(df, pd.DataFrame):
                return df.copy()
        except Exception:
            pass

    raise TypeError(
        "Non trovo una tabella eventi leggibile nell'OCEL "
        "(attesi ad es. ocel.log.log, ocel.log, ocel.events o get_extended_table())."
    )


def _infer_event_id_col(df: pd.DataFrame) -> str:
    candidates = [
        "ocel:eid",
        "event_id",
        "event id",
        "eventid",
        "eid",
    ]
    for c in candidates:
        if c in df.columns:
            return c

    raise ValueError(
        "Non trovo la colonna id evento nella tabella del log. "
        f"Colonne disponibili: {list(df.columns)}"
    )


def _get_explicit_objects_df(ocel) -> pd.DataFrame | None:
    candidates = [
        getattr(ocel, "objects", None),
        getattr(getattr(ocel, "log", None), "objects", None),
        getattr(getattr(getattr(ocel, "log", None), "log", None), "objects", None),
    ]
    for obj in candidates:
        if isinstance(obj, pd.DataFrame):
            return obj.copy()
    return None


def _get_explicit_relations_df(ocel) -> pd.DataFrame | None:
    candidates = [
        getattr(ocel, "relations", None),
        getattr(getattr(ocel, "log", None), "relations", None),
        getattr(getattr(getattr(ocel, "log", None), "log", None), "relations", None),
    ]
    for obj in candidates:
        if isinstance(obj, pd.DataFrame):
            return obj.copy()
    return None


# ============================================================
# Type graph
# ============================================================

def _normalize_type_links(type_links: Dict[str, List[str] | str]) -> Dict[str, List[str]]:
    out = {}
    for src, tgts in type_links.items():
        src = str(src)
        if isinstance(tgts, str):
            tgts = [tgts]
        out[src] = [str(t) for t in tgts]
    return out


def _all_nodes(type_links: Dict[str, List[str]]) -> Set[str]:
    nodes = set(type_links.keys())
    for tgts in type_links.values():
        nodes.update(tgts)
    return nodes


def _in_degrees(type_links: Dict[str, List[str]]) -> Dict[str, int]:
    nodes = _all_nodes(type_links)
    indeg = {n: 0 for n in nodes}
    for src, tgts in type_links.items():
        for tgt in tgts:
            indeg[tgt] += 1
    return indeg


def _roots(type_links: Dict[str, List[str]]) -> List[str]:
    indeg = _in_degrees(type_links)
    return sorted([n for n, d in indeg.items() if d == 0])


def _reachable_from(root: str, graph: Dict[str, List[str]]) -> Set[str]:
    seen = set()
    stack = [root]

    while stack:
        u = stack.pop()
        if u in seen:
            continue
        seen.add(u)
        for v in graph.get(u, []):
            stack.append(v)

    return seen


def _topological_sort(graph: Dict[str, List[str]]) -> List[str]:
    nodes = _all_nodes(graph)
    indeg = {n: 0 for n in nodes}

    for u, vs in graph.items():
        for v in vs:
            indeg[v] += 1

    q = deque(sorted([n for n, d in indeg.items() if d == 0]))
    order = []

    while q:
        u = q.popleft()
        order.append(u)
        for v in graph.get(u, []):
            indeg[v] -= 1
            if indeg[v] == 0:
                q.append(v)

    if len(order) != len(nodes):
        raise ValueError("Il grafo dei tipi contiene un ciclo. Serve un DAG aciclico.")

    return order


def _maximal_nodes(reachable: Set[str], graph: Dict[str, List[str]]) -> List[str]:
    """
    Maximal nodes in the reachable subgraph:
    no outgoing edge towards another reachable node.
    """
    out = []
    for u in reachable:
        succs = [v for v in graph.get(u, []) if v in reachable]
        if not succs:
            out.append(u)
    return sorted(out)


# ============================================================
# Object and relation reconstruction
# ============================================================

def _get_object_ids_by_type_any(
    ocel,
    known_object_types: List[str],
) -> Dict[str, List[str]]:
    """
    Restituisce:
        tipo -> lista ordinata di object ids

    First tries with an explicit object table.
    Altrimenti ricostruisce dalla tabella larga.
    """
    obj_df = _get_explicit_objects_df(ocel)

    if obj_df is not None and OBJ_ID in obj_df.columns and OBJ_TYPE in obj_df.columns:
        tmp = obj_df[[OBJ_ID, OBJ_TYPE]].copy()
        tmp[OBJ_ID] = tmp[OBJ_ID].astype(str)
        tmp[OBJ_TYPE] = tmp[OBJ_TYPE].astype(str)

        out = {}
        for ot, g in tmp.groupby(OBJ_TYPE):
            out[str(ot)] = sorted(g[OBJ_ID].drop_duplicates().tolist())
        return out

    log_df = _get_log_df(ocel)

    missing_cols = [ot for ot in known_object_types if ot not in log_df.columns]
    if missing_cols:
        raise ValueError(
            "Non trovo queste colonne di object type nella tabella del log: "
            f"{missing_cols}. Colonne disponibili: {list(log_df.columns)}"
        )

    out = {ot: set() for ot in known_object_types}

    for ot in known_object_types:
        for val in log_df[ot]:
            out[ot].update(_norm_cell(val))

    return {ot: sorted(vals) for ot, vals in out.items()}


def _build_event_index_any(
    ocel,
    known_object_types: List[str],
) -> Dict[str, Dict[str, Set[str]]]:
    """
    Restituisce:
        event_index[eid][otype] = {oid1, oid2, ...}

    Usa relations esplicite se presenti, altrimenti ricostruisce
    dalla tabella larga.
    """
    rel_df = _get_explicit_relations_df(ocel)

    if rel_df is not None and {EV_ID, OBJ_ID, OBJ_TYPE}.issubset(rel_df.columns):
        tmp = rel_df[[EV_ID, OBJ_ID, OBJ_TYPE]].copy()
        tmp[EV_ID] = tmp[EV_ID].astype(str)
        tmp[OBJ_ID] = tmp[OBJ_ID].astype(str)
        tmp[OBJ_TYPE] = tmp[OBJ_TYPE].astype(str)

        event_index = defaultdict(lambda: defaultdict(set))
        for _, row in tmp.iterrows():
            event_index[row[EV_ID]][row[OBJ_TYPE]].add(row[OBJ_ID])

        return event_index

    log_df = _get_log_df(ocel)
    eid_col = _infer_event_id_col(log_df)

    missing_cols = [ot for ot in known_object_types if ot not in log_df.columns]
    if missing_cols:
        raise ValueError(
            "Non trovo queste colonne di object type nella tabella del log: "
            f"{missing_cols}. Colonne disponibili: {list(log_df.columns)}"
        )

    event_index = defaultdict(lambda: defaultdict(set))

    for _, row in log_df.iterrows():
        eid = str(row[eid_col])
        for ot in known_object_types:
            event_index[eid][ot].update(_norm_cell(row[ot]))

    return event_index


# ============================================================
# Direct mapping between types
# ============================================================

def _infer_direct_mapping(
    event_index: Dict[str, Dict[str, Set[str]]],
    source_type: str,
    target_type: str,
    source_object_ids: List[str],
    mode: str = "strict",
) -> Dict[str, str]:
    """
    Ricava un mapping funzionale:
        source_obj -> target_obj

    guardando la co-occorrenza negli stessi eventi.

    mode:
    - "strict": if a source co-occurs with multiple distinct targets, raise error
    - "most_frequent": pick the most frequent target
    """
    cooc = defaultdict(list)

    for _, type_map in event_index.items():
        srcs = type_map.get(source_type, set())
        tgts = type_map.get(target_type, set())

        if not srcs or not tgts:
            continue

        for s in srcs:
            for t in tgts:
                cooc[s].append(t)

    mapping = {}
    missing = []
    ambiguous = {}

    for s in source_object_ids:
        tgts = cooc.get(s, [])
        if not tgts:
            missing.append(s)
            continue

        counts = Counter(tgts)

        if len(counts) == 1:
            mapping[s] = next(iter(counts))
        else:
            if mode == "most_frequent":
                mapping[s] = counts.most_common(1)[0][0]
            else:
                ambiguous[s] = dict(counts)

    if missing:
        raise ValueError(
            f"Impossibile ricavare il mapping {source_type} -> {target_type} "
            f"for {len(missing)} objects. Examples: {missing[:10]}"
        )

    if ambiguous:
        ex = list(ambiguous.items())[:5]
        raise ValueError(
            f"The mapping {source_type} -> {target_type} is not unique. "
            f"Esempi: {ex}"
        )

    return mapping


def _build_all_direct_mappings(
    ocel,
    type_links: Dict[str, List[str]],
    mode: str = "strict",
) -> Dict[Tuple[str, str], Dict[str, str]]:
    """
    Costruisce tutti i mapping diretti richiesti dagli archi del grafo.
    """
    all_types = sorted(_all_nodes(type_links))
    event_index = _build_event_index_any(ocel, all_types)
    object_ids_by_type = _get_object_ids_by_type_any(ocel, all_types)

    direct = {}

    for src, tgts in type_links.items():
        if src not in object_ids_by_type:
            raise ValueError(f"Type '{src}' not found in log objects.")

        for tgt in tgts:
            if tgt not in object_ids_by_type:
                raise ValueError(f"Type '{tgt}' not found in log objects.")

            direct[(src, tgt)] = _infer_direct_mapping(
                event_index=event_index,
                source_type=src,
                target_type=tgt,
                source_object_ids=object_ids_by_type[src],
                mode=mode,
            )

    return direct


# ============================================================
# Composizione mapping
# ============================================================

def _compose_maps(
    map_a_to_b: Dict[str, str],
    map_b_to_c: Dict[str, str],
) -> Dict[str, str]:
    """
    Compone:
        a -> b
        b -> c
    ottenendo:
        a -> c
    """
    out = {}
    missing_mid = []

    for a, b in map_a_to_b.items():
        if b not in map_b_to_c:
            missing_mid.append((a, b))
            continue
        out[a] = map_b_to_c[b]

    if missing_mid:
        raise ValueError(
            "Composition failed: some intermediate objects have no successor mapping. "
            f"Esempi: {missing_mid[:10]}"
        )

    return out


def _merge_candidate_maps(
    candidate_maps: List[Dict[str, str]],
    root_object_ids: List[str],
    mode: str = "strict",
) -> Dict[str, str]:
    """
    Merges multiple root -> node maps produced by different paths.
    Se per uno stesso oggetto radice emergono target diversi:
    - strict -> raise error
    - most_frequent -> pick the most frequent
    """
    bucket = defaultdict(list)
    for cmap in candidate_maps:
        for r, x in cmap.items():
            bucket[r].append(x)

    out = {}
    conflicts = {}
    missing = []

    for r in root_object_ids:
        vals = bucket.get(r, [])
        if not vals:
            missing.append(r)
            continue

        counts = Counter(vals)

        if len(counts) == 1:
            out[r] = next(iter(counts))
        else:
            if mode == "most_frequent":
                out[r] = counts.most_common(1)[0][0]
            else:
                conflicts[r] = dict(counts)

    if missing:
        raise ValueError(
            "Some root objects do not reach this node. "
            f"Esempi: {missing[:10]}"
        )

    if conflicts:
        ex = list(conflicts.items())[:5]
        raise ValueError(
            "Cammini diversi nel lattice producono mapping incoerenti. "
            f"Esempi: {ex}"
        )

    return out


# ============================================================
# Costruzione JSON finale
# ============================================================
def build_service_hierarchy_json(
    ocel,
    type_links,
    *,
    sqlite_path: str,
    mode: str = "strict",
):
    rooted = build_rooted_lattice_json(
        ocel=ocel,
        type_links=type_links,
        sqlite_path=sqlite_path,
        mode=mode,
    )

    out = {}

    for tau, block in rooted.items():
        specs = []

        for node_name, node_spec in block["nodes"].items():
            if node_name == block["top"]:
                continue

            specs.append({
                "name": node_name,
                "mapping": node_spec["mapping"]
            })

        out[tau] = specs

    return out

import json
from pathlib import Path

def save_service_hierarchy_json(hierarchy_json, out_path):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(hierarchy_json, f, indent=2, ensure_ascii=False)

def build_rooted_lattice_json(
    ocel,
    type_links: Dict[str, List[str] | str],
    *,
    sqlite_path: str,
    mode: str = "strict",
) -> Dict[str, Any]:
    """
    Costruisce un lattice per ogni tipo radice.

    Esempio:
        type_links = {
            "packages": ["items"],
            "items": ["product"],
            "orders": ["customers"],
        }

    Output concettuale:
        packages -> items -> product -> packages_all
        orders   -> customers -> orders_all

    Within a root's block, every mapping is always expressed as:
        oggetto_della_radice -> oggetto_del_nodo
    """
    if mode not in {"strict", "most_frequent"}:
        raise ValueError("mode deve essere 'strict' oppure 'most_frequent'.")

    graph = _normalize_type_links(type_links)
    topo = _topological_sort(graph)
    roots = _roots(graph)

    if not roots:
        raise ValueError("Non ho trovato tipi radice nel grafo dei type_links.")

    direct_maps, object_ids_by_type = _build_all_direct_mappings_from_sqlite(
        sqlite_path=sqlite_path,
        type_links=graph,
        mode=mode,
    )

    result = {}

    for root in roots:
        if root not in object_ids_by_type:
            raise ValueError(f"Root '{root}' not found in log objects.")

        reachable = _reachable_from(root, graph)
        sub_topo = [n for n in topo if n in reachable]
        root_object_ids = object_ids_by_type[root]

        if not root_object_ids:
            raise ValueError(f"No objects found for root '{root}'.")

        # Per ogni nodo raggiungibile mantengo una mappa:
        #   root_obj -> node_obj
        root_to_node = {
            root: {oid: oid for oid in root_object_ids}
        }

        for node in sub_topo:
            if node == root:
                continue

            preds = [
                p for p in reachable
                if node in graph.get(p, []) and p in root_to_node
            ]

            candidate_maps = []
            for p in preds:
                root_to_p = root_to_node[p]
                p_to_node = direct_maps[(p, node)]
                candidate_maps.append(_compose_maps(root_to_p, p_to_node))

            if not candidate_maps:
                raise ValueError(
                    f"Node '{node}' is reachable from '{root}', "
                    "but cannot build the mapping from the root."
                )

            root_to_node[node] = _merge_candidate_maps(
                candidate_maps=candidate_maps,
                root_object_ids=root_object_ids,
                mode=mode,
            )

        top_name = f"{root}_all"
        star_name = f"{root}_star"

        nodes = {}
        for node in sub_topo:
            if node == root:
                continue
            nodes[node] = {"mapping": root_to_node[node]}

        nodes[top_name] = {
            "mapping": {oid: star_name for oid in root_object_ids}
        }

        lattice_edges = []
        for u in sub_topo:
            for v in graph.get(u, []):
                if v in reachable:
                    lattice_edges.append([u, v])

        for max_node in _maximal_nodes(reachable, graph):
            lattice_edges.append([max_node, top_name])

        result[root] = {
            "bottom": root,
            "top": top_name,
            "nodes": nodes,
            "lattice_edges": lattice_edges,
        }

    return result


def save_lattice_json(lattice_json: Dict[str, Any], out_path: str | Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(lattice_json, f, indent=2, ensure_ascii=False)

import os
import tempfile
from pathlib import Path
import pm4py


def build_filtered_logs_pm4py_and_ocpa(
    input_log,
    source_format,   # "pm4py" oppure "ocpa"
    keep_event_types,
    keep_object_types,
    resource_object_types,
    output_path=None,                     # <-- meglio nome generico
    resource_prefix="resource_",
    resource_encoding="json_list",
    joined_sep="|",
    keep_events_without_kept_objects=True,
    activity_object_rules=None
):
    """
    Restituisce:
        filtered_pm4py_ocel
        filtered_ocpa_ocel

    If output_path is set, saves the filtered log to disk.
    Supporta sia .jsonocel/.json sia .sqlite.
    """

    source_format = source_format.lower().strip()
    if source_format not in {"pm4py", "ocpa"}:
        raise ValueError("source_format deve essere 'pm4py' oppure 'ocpa'.")

    from ocpa.objects.log.importer.ocel import factory as ocpa_import_factory_json
    from ocpa.objects.log.importer.ocel2.sqlite import factory as ocpa_import_factory_sqlite
    from ocpa.objects.log.exporter.ocel import factory as ocpa_export_factory

    temp_input = None
    temp_output = None

    try:
        # A) porta tutto in PM4Py
        if source_format == "pm4py":
            pm4py_input = input_log
        else:
            fd_in, temp_input = tempfile.mkstemp(suffix=".jsonocel")
            os.close(fd_in)
            ocpa_export_factory.apply(input_log, temp_input)
            pm4py_input = pm4py.read_ocel(temp_input)

        # B) costruisci il nuovo PM4Py OCEL
        filtered_pm4py = build_filtered_pm4py_ocel_with_resources(
            ocel=pm4py_input,
            keep_event_types=keep_event_types,
            keep_object_types=keep_object_types,
            resource_object_types=resource_object_types,
            resource_prefix=resource_prefix,
            resource_encoding=resource_encoding,
            joined_sep=joined_sep,
            keep_events_without_kept_objects=keep_events_without_kept_objects,
        )

        # C) apply optional activity-object rules
        filtered_pm4py = _apply_activity_object_rules_to_pm4py_ocel(
            ocel=filtered_pm4py,
            activity_object_rules=activity_object_rules,
            keep_events_without_kept_objects=keep_events_without_kept_objects,
        )

        # D) scegli path di output
        if output_path is None:
            fd_out, temp_output = tempfile.mkstemp(suffix=".jsonocel")
            os.close(fd_out)
            output_path = temp_output

        # E) save
        pm4py.write_ocel(filtered_pm4py, output_path)

        if not os.path.exists(output_path):
            raise FileNotFoundError(
                f"PM4Py did not create the expected file: {output_path}"
            )

        # F) reimporta in OCPA in base al formato
        ext = Path(output_path).suffix.lower()

        if ext == ".sqlite":
            filtered_ocpa = ocpa_import_factory_sqlite.apply(output_path)
        elif ext in {".jsonocel", ".json"}:
            filtered_ocpa = ocpa_import_factory_json.apply(output_path)
        else:
            raise ValueError(
                f"Unsupported output format: {output_path}. "
                f"Usa .jsonocel/.json oppure .sqlite"
            )

        return filtered_pm4py, filtered_ocpa

    finally:
        if temp_input and os.path.exists(temp_input):
            os.remove(temp_input)

        if temp_output and os.path.exists(temp_output):
            os.remove(temp_output)


import copy
import json
import os
import tempfile
import pandas as pd
import pm4py
from pm4py.objects.ocel.obj import OCEL


def _apply_activity_object_rules_to_pm4py_ocel(
    ocel,
    activity_object_rules: dict[str, set[str]] | None,
    keep_events_without_kept_objects: bool = True,
):
    """
    Applica regole del tipo:
        activity -> insieme di object types ammessi

    Agisce solo sulle relazioni evento-oggetto.
    """
    if not activity_object_rules:
        return ocel

    new_ocel = copy.deepcopy(ocel)

    events_df = new_ocel.events.copy()
    objects_df = new_ocel.objects.copy()
    relations_df = new_ocel.relations.copy()

    # event columns
    ev_id_col = _pick_col(events_df, ["ocel:eid", "event_id", "eid"])
    ev_act_col = _pick_col(events_df, ["ocel:activity", "event_activity", "activity"])

    # object columns
    obj_id_col = _pick_col(objects_df, ["ocel:oid", "object_id", "oid"])
    obj_type_col = _pick_col(objects_df, ["ocel:type", "object_type", "type"])

    # relation columns
    rel_ev_col = _pick_col(relations_df, [ev_id_col, "ocel:eid", "event_id", "eid"])
    rel_obj_col = _pick_col(relations_df, [obj_id_col, "ocel:oid", "object_id", "oid"])

    rel = relations_df.copy()

    # activity nella relations
    if ev_act_col in rel.columns:
        rel_act_col = ev_act_col
    else:
        rel_act_col = "__tmp_activity__"
        ev_map = (
            events_df[[ev_id_col, ev_act_col]]
            .drop_duplicates()
            .rename(columns={ev_id_col: rel_ev_col, ev_act_col: rel_act_col})
        )
        rel = rel.merge(ev_map, on=rel_ev_col, how="left", validate="many_to_one")

    # object type nella relations
    if obj_type_col in rel.columns:
        rel_type_col = obj_type_col
    else:
        rel_type_col = "__tmp_obj_type__"
        obj_map = (
            objects_df[[obj_id_col, obj_type_col]]
            .drop_duplicates()
            .rename(columns={obj_id_col: rel_obj_col, obj_type_col: rel_type_col})
        )
        rel = rel.merge(obj_map, on=rel_obj_col, how="left", validate="many_to_one")

    mask_keep = rel.apply(
        lambda row: (
            True
            if row[rel_act_col] not in activity_object_rules
            else row[rel_type_col] in activity_object_rules[row[rel_act_col]]
        ),
        axis=1,
    )

    filtered_relations_df = relations_df.loc[mask_keep.values].copy()

    # clean up objects with no remaining links
    kept_object_ids = set(filtered_relations_df[rel_obj_col].dropna().unique())
    filtered_objects_df = objects_df[
        objects_df[obj_id_col].isin(kept_object_ids)
    ].copy()

    # optional: remove events left with no objects
    if keep_events_without_kept_objects:
        filtered_events_df = events_df.copy()
    else:
        kept_event_ids = set(filtered_relations_df[rel_ev_col].dropna().unique())
        filtered_events_df = events_df[
            events_df[ev_id_col].isin(kept_event_ids)
        ].copy()

    new_ocel.events = filtered_events_df
    new_ocel.objects = filtered_objects_df
    new_ocel.relations = filtered_relations_df

    return new_ocel

def _encode_resource_values(values, encoding="json_list", sep="|"):
    vals = sorted({str(v) for v in values if pd.notna(v)})

    if encoding == "json_list":
        return json.dumps(vals, ensure_ascii=False)

    if encoding == "joined":
        return sep.join(vals)

    if encoding == "single":
        if len(vals) > 1:
            raise ValueError(
                f"Multiple resources found where a single value was expected: {vals}"
            )
        return vals[0] if vals else None

    raise ValueError(f"Unsupported resource_encoding: {encoding}")


def build_filtered_pm4py_ocel_with_resources(
    ocel: OCEL,
    keep_event_types,
    keep_object_types,
    resource_object_types,
    resource_prefix="resource_",
    resource_encoding="json_list",   # "json_list" | "joined" | "single"
    joined_sep="|",
    keep_events_without_kept_objects=True,
):
    """
    Crea un NUOVO PM4Py OCEL:
    - tiene solo alcuni tipi di evento
    - tiene solo alcuni tipi di oggetto come object types veri
    - converte altri object types in attributi evento (risorse)

    Ritorna:
        new_ocel : pm4py.objects.ocel.obj.OCEL
    """

    keep_event_types = set(keep_event_types)
    keep_object_types = set(keep_object_types)
    resource_object_types = set(resource_object_types)

    overlap = keep_object_types & resource_object_types
    if overlap:
        raise ValueError(
            f"These types appear both as kept objects and as resources: {sorted(overlap)}"
        )

    EV_ID = ocel.event_id_column
    ACTIVITY = ocel.event_activity
    TIMESTAMP = ocel.event_timestamp
    OBJ_ID = ocel.object_id_column
    OBJ_TYPE = ocel.object_type_column

    events = ocel.events.copy()
    objects = ocel.objects.copy()
    relations = ocel.relations.copy()

    # 1) filter events
    new_events = events[events[ACTIVITY].isin(keep_event_types)].copy()
    kept_eids = set(new_events[EV_ID])

    # 2) relations for kept events only
    rel = relations[relations[EV_ID].isin(kept_eids)].copy()

    # 3) add object type in a working view
    #    senza rompere lo schema originale di relations
    rel_work = rel.merge(
        objects[[OBJ_ID, OBJ_TYPE]],
        on=OBJ_ID,
        how="left",
        suffixes=("", "__obj")
    )

    obj_type_work_col = OBJ_TYPE if OBJ_TYPE in rel_work.columns else f"{OBJ_TYPE}__obj"
    if obj_type_work_col not in rel_work.columns:
        # if OBJ_TYPE was already in rel, after merge it stays;
        # altrimenti dovrebbe esserci OBJ_TYPE__obj
        fallback = f"{OBJ_TYPE}__obj"
        if fallback in rel_work.columns:
            obj_type_work_col = fallback
        else:
            raise ValueError("Impossibile recuperare il tipo oggetto nelle relazioni.")

    # 4) costruisci attributi evento dalle risorse
    rel_resources = rel_work[rel_work[obj_type_work_col].isin(resource_object_types)].copy()

    if not rel_resources.empty:
        grouped = (
            rel_resources
            .groupby([EV_ID, obj_type_work_col])[OBJ_ID]
            .agg(list)
            .reset_index()
        )

        for ot in sorted(resource_object_types):
            tmp = grouped[grouped[obj_type_work_col] == ot][[EV_ID, OBJ_ID]].copy()
            if tmp.empty:
                continue

            tmp[f"{resource_prefix}{ot}"] = tmp[OBJ_ID].apply(
                lambda x: _encode_resource_values(
                    x,
                    encoding=resource_encoding,
                    sep=joined_sep
                )
            )
            tmp = tmp[[EV_ID, f"{resource_prefix}{ot}"]]

            new_events = new_events.merge(tmp, on=EV_ID, how="left")

    # 5) filtra object table
    new_objects = objects[objects[OBJ_TYPE].isin(keep_object_types)].copy()
    kept_oids = set(new_objects[OBJ_ID])

    # 6) filter relations to kept objects only
    new_relations = rel[rel[OBJ_ID].isin(kept_oids)].copy()

    # 7) optional: drop events left with no real objects
    if not keep_events_without_kept_objects:
        eids_with_objects = set(new_relations[EV_ID])
        new_events = new_events[new_events[EV_ID].isin(eids_with_objects)].copy()
        kept_eids = set(new_events[EV_ID])
        new_relations = new_relations[new_relations[EV_ID].isin(kept_eids)].copy()

    # 8) clean up indices
    new_events = new_events.reset_index(drop=True)
    new_objects = new_objects.reset_index(drop=True)
    new_relations = new_relations.reset_index(drop=True)

    # 9) ricostruisci il nuovo OCEL PM4Py
    new_ocel = OCEL(
        events=new_events,
        objects=new_objects,
        relations=new_relations,
        globals=copy.deepcopy(getattr(ocel, "globals", {})),
        parameters=copy.deepcopy(getattr(ocel, "parameters", {})),
    )

    return new_ocel