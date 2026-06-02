from __future__ import annotations

import copy
import json
import re
import sqlite3
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path
from typing import Any

def safe_name(name: str) -> str:
    name = str(name)
    name = re.sub('[^A-Za-z0-9_]+', '_', name)
    name = re.sub('_+', '_', name).strip('_')
    if not name:
        name = 'unnamed'
    if name[0].isdigit():
        name = 'x_' + name
    return name.lower()

def compact_o2o_relations(cards):
    compact = []
    seen = set()
    for (a, b), info_ab in sorted(cards.items()):
        if (b, a) in seen or a == b:
            continue
        seen.add((a, b))
        info_ba = cards.get((b, a))
        if info_ba is None:
            continue
        if info_ab['max'] == 0 and info_ba['max'] == 0:
            continue
        compact.append({'left': a, 'right': b, 'a_to_b_min': info_ab['min'], 'a_to_b_max': info_ab['max'], 'a_to_b_range': info_ab['range'], 'b_to_a_min': info_ba['min'], 'b_to_a_max': info_ba['max'], 'b_to_a_range': info_ba['range']})
    return compact

def range_label(min_c: int, max_c: int) -> str:
    if max_c == 0:
        return '0'
    if min_c == 0 and max_c == 1:
        return '0..1'
    if min_c == 1 and max_c == 1:
        return '1'
    if min_c >= 1 and max_c > 1:
        return '1..*'
    return '0..*'

def o2o_cardinalities_from_sqlite(sqlite_path: str, include_same_type: bool=False, include_zeros: bool=True):
    con = sqlite3.connect(sqlite_path)
    cur = con.cursor()
    obj_rows = cur.execute('\n        SELECT ocel_id, ocel_type\n        FROM object\n    ').fetchall()
    object_to_type = {}
    type_to_objects = defaultdict(set)
    for obj_id, obj_type in obj_rows:
        object_to_type[obj_id] = obj_type
        type_to_objects[obj_type].add(obj_id)
    o2o_rows = cur.execute('\n        SELECT ocel_source_id, ocel_target_id\n        FROM object_object\n    ').fetchall()
    con.close()
    neighbors = defaultdict(lambda: defaultdict(set))
    for src, tgt in o2o_rows:
        if src not in object_to_type or tgt not in object_to_type:
            continue
        src_type = object_to_type[src]
        tgt_type = object_to_type[tgt]
        if not include_same_type and src_type == tgt_type:
            continue
        neighbors[src_type, tgt_type][src].add(tgt)
        neighbors[tgt_type, src_type][tgt].add(src)
    results = {}
    all_types = sorted(type_to_objects.keys())
    for type_a in all_types:
        for type_b in all_types:
            if not include_same_type and type_a == type_b:
                continue
            source_objects = type_to_objects[type_a]
            if not source_objects:
                continue
            if include_zeros:
                counts = [len(neighbors[type_a, type_b].get(obj_a, set())) for obj_a in source_objects]
            else:
                counts = [len(neighbors[type_a, type_b][obj_a]) for obj_a in neighbors[type_a, type_b] if len(neighbors[type_a, type_b][obj_a]) > 0]
            if not counts:
                continue
            min_c = min(counts)
            max_c = max(counts)
            results[type_a, type_b] = {'min': min_c, 'max': max_c, 'range': range_label(min_c, max_c), 'num_objects_source': len(source_objects), 'num_objects_with_link': sum((1 for c in counts if c > 0))}
    return (results, o2o_rows)

def o2o_cardinalities_from_log(log: dict, include_same_type: bool=False, include_zeros: bool=True):
    object_to_type = {oid: obj['type'] for oid, obj in log['objects'].items()}
    type_to_objects = defaultdict(set)
    for oid, obj_type in object_to_type.items():
        type_to_objects[obj_type].add(oid)
    o2o_rows = []
    for r in log.get('o2o', []):
        src = r.get('source')
        tgt = r.get('target')
        if src is not None and tgt is not None:
            o2o_rows.append((src, tgt))
    neighbors = defaultdict(lambda: defaultdict(set))
    for src, tgt in o2o_rows:
        if src not in object_to_type or tgt not in object_to_type:
            continue
        src_type = object_to_type[src]
        tgt_type = object_to_type[tgt]
        if not include_same_type and src_type == tgt_type:
            continue
        neighbors[src_type, tgt_type][src].add(tgt)
        neighbors[tgt_type, src_type][tgt].add(src)
    results = {}
    all_types = sorted(type_to_objects.keys())
    for type_a in all_types:
        for type_b in all_types:
            if not include_same_type and type_a == type_b:
                continue
            source_objects = type_to_objects[type_a]
            if not source_objects:
                continue
            if include_zeros:
                counts = [len(neighbors[type_a, type_b].get(obj_a, set())) for obj_a in source_objects]
            else:
                counts = [len(neighbors[type_a, type_b][obj_a]) for obj_a in neighbors[type_a, type_b] if len(neighbors[type_a, type_b][obj_a]) > 0]
            if not counts:
                continue
            min_c = min(counts)
            max_c = max(counts)
            results[type_a, type_b] = {'min': min_c, 'max': max_c, 'range': range_label(min_c, max_c), 'num_objects_source': len(source_objects), 'num_objects_with_link': sum((1 for c in counts if c > 0))}
    return (results, o2o_rows)

def load_ocel(path: str | Path) -> dict:
    p = Path(path)
    suffix = p.suffix.lower()
    if suffix in ('.sqlite', '.db'):
        return _load_sqlite(p)
    elif suffix == '.xml':
        return _load_xml(p)
    elif suffix in ('.json', '.jsonocel'):
        raw = json.loads(p.read_text(encoding='utf-8'))
        if 'ocel:global-log' in raw or 'ocel:global-event' in raw:
            return _load_ocel1_json(raw)
        else:
            return _load_ocel2_json(raw)
    else:
        raise ValueError(f'Formato non riconosciuto: {suffix}')

def export_ocel(log: dict, path: str | Path) -> None:
    fmt = log['format']
    p = Path(path)
    dispatch = {'ocel1_json': _export_ocel1_json, 'ocel2_json': _export_ocel2_json, 'ocel2_xml': _export_ocel2_xml, 'ocel2_sqlite': _export_ocel2_sqlite}
    if fmt not in dispatch:
        raise ValueError(f'Formato sconosciuto: {fmt}')
    dispatch[fmt](log, p)
    print(f'[export] Log salvato in: {p}')

def _load_ocel1_json(raw: dict) -> dict:
    events = {}
    for eid, ev in raw.get('ocel:events', {}).items():
        events[eid] = {'activity': ev.get('ocel:activity', ''), 'timestamp': ev.get('ocel:timestamp', ''), 'attributes': {k: v for k, v in ev.items() if k not in ('ocel:activity', 'ocel:timestamp', 'ocel:omap', 'ocel:vmap')}, 'objects': list(ev.get('ocel:omap', []))}
        events[eid]['attributes'].update(ev.get('ocel:vmap', {}))
    objects = {}
    for oid, obj in raw.get('ocel:objects', {}).items():
        objects[oid] = {'type': obj.get('ocel:type', ''), 'attributes': dict(obj.get('ocel:ovmap', {}))}
    o2o = [{'source': r['ocel:source-id'], 'target': r['ocel:target-id'], 'qualifier': r.get('ocel:qualifier', '')} for r in raw.get('ocel:o2o', [])]
    return {'events': events, 'objects': objects, 'o2o': o2o, 'format': 'ocel1_json', '_raw': raw}

def _export_ocel1_json(log: dict, path: Path) -> None:
    raw = copy.deepcopy(log['_raw'])
    for eid, ev in log['events'].items():
        raw_ev = raw['ocel:events'].setdefault(eid, {})
        raw_ev['ocel:activity'] = ev['activity']
        raw_ev['ocel:timestamp'] = ev['timestamp']
        raw_ev['ocel:omap'] = ev['objects']
        raw_ev['ocel:vmap'] = dict(ev['attributes'])
    for eid in list(raw['ocel:events']):
        if eid not in log['events']:
            del raw['ocel:events'][eid]
    raw['ocel:objects'] = {oid: {'ocel:type': obj['type'], 'ocel:ovmap': obj['attributes']} for oid, obj in log['objects'].items()}
    raw['ocel:o2o'] = [{'ocel:source-id': r['source'], 'ocel:target-id': r['target'], 'ocel:qualifier': r['qualifier']} for r in log['o2o']]
    path.write_text(json.dumps(raw, indent=2, ensure_ascii=False), encoding='utf-8')

def _load_ocel2_json(raw: dict) -> dict:
    objects = {}
    for obj in raw.get('objects', []):
        attrs = {a['name']: a['value'] for a in obj.get('attributes', [])}
        objects[obj['id']] = {'type': obj['type'], 'attributes': attrs}
    events = {}
    for ev in raw.get('events', []):
        attrs = {a['name']: a['value'] for a in ev.get('attributes', [])}
        objs = [r['objectId'] for r in ev.get('relationships', [])]
        events[ev['id']] = {'activity': ev.get('type', ''), 'timestamp': ev.get('time', ''), 'attributes': attrs, 'objects': objs}
    o2o = []
    for obj in raw.get('objects', []):
        for rel in obj.get('relationships', []):
            o2o.append({'source': obj['id'], 'target': rel['objectId'], 'qualifier': rel.get('qualifier', '')})
    return {'events': events, 'objects': objects, 'o2o': o2o, 'format': 'ocel2_json', '_raw': raw}

def _export_ocel2_json(log: dict, path: Path) -> None:
    obj_list = []
    for oid, obj in log['objects'].items():
        rels = [{'objectId': r['target'], 'qualifier': r['qualifier']} for r in log['o2o'] if r['source'] == oid]
        obj_list.append({'id': oid, 'type': obj['type'], 'attributes': [{'name': k, 'value': v} for k, v in obj['attributes'].items()], 'relationships': rels})
    ev_list = []
    for eid, ev in log['events'].items():
        ev_list.append({'id': eid, 'type': ev['activity'], 'time': ev['timestamp'], 'attributes': [{'name': k, 'value': v} for k, v in ev['attributes'].items()], 'relationships': [{'objectId': oid, 'qualifier': ''} for oid in ev['objects']]})
    raw = copy.deepcopy(log['_raw'])
    raw['objects'] = obj_list
    raw['events'] = ev_list
    path.write_text(json.dumps(raw, indent=2, ensure_ascii=False), encoding='utf-8')

def _load_xml(path: Path) -> dict:
    tree = ET.parse(path)
    root = tree.getroot()
    ns = {'ocel': 'http://www.ocel-standard.org/'}

    def findall(node, tag):
        r = node.findall(f'ocel:{tag}', ns)
        return r if r else node.findall(tag)
    objects = {}
    for obj in findall(root, 'object'):
        oid = obj.get('id') or obj.get('ocel:id')
        otype = obj.get('type') or obj.get('ocel:type')
        attrs = {}
        for a in findall(obj, 'attribute'):
            attrs[a.get('name') or a.get('ocel:name')] = a.text
        objects[oid] = {'type': otype, 'attributes': attrs}
    events = {}
    for ev in findall(root, 'event'):
        eid = ev.get('id') or ev.get('ocel:id')
        act = ev.get('activity') or ev.get('ocel:activity') or ev.get('type') or ''
        ts = ev.get('timestamp') or ev.get('ocel:timestamp') or ev.get('time') or ''
        attrs = {}
        for a in findall(ev, 'attribute'):
            attrs[a.get('name') or a.get('ocel:name')] = a.text
        objs = [r.get('objectId') or r.get('ocel:objectId') for r in findall(ev, 'relationship')]
        events[eid] = {'activity': act, 'timestamp': ts, 'attributes': attrs, 'objects': objs}
    o2o = []
    for obj in findall(root, 'object'):
        oid = obj.get('id') or obj.get('ocel:id')
        for r in findall(obj, 'relationship'):
            o2o.append({'source': oid, 'target': r.get('objectId') or r.get('ocel:objectId'), 'qualifier': r.get('qualifier', '')})
    return {'events': events, 'objects': objects, 'o2o': o2o, 'format': 'ocel2_xml', '_raw': ET.tostring(root, encoding='unicode')}

def _export_ocel2_xml(log: dict, path: Path) -> None:
    root = ET.Element('log')
    for oid, obj in log['objects'].items():
        o_el = ET.SubElement(root, 'object', id=oid, type=obj['type'])
        for k, v in obj['attributes'].items():
            a_el = ET.SubElement(o_el, 'attribute', name=k)
            a_el.text = str(v)
        for r in log['o2o']:
            if r['source'] == oid:
                ET.SubElement(o_el, 'relationship', objectId=r['target'], qualifier=r['qualifier'])
    for eid, ev in log['events'].items():
        e_el = ET.SubElement(root, 'event', id=eid, activity=ev['activity'], timestamp=str(ev['timestamp']))
        for k, v in ev['attributes'].items():
            a_el = ET.SubElement(e_el, 'attribute', name=k)
            a_el.text = str(v)
        for oid in ev['objects']:
            ET.SubElement(e_el, 'relationship', objectId=oid, qualifier='')
    ET.indent(root, space='  ')
    ET.ElementTree(root).write(path, encoding='utf-8', xml_declaration=True)

def _normalize_object_attributes(objects: dict[str, dict]) -> None:
    for obj in objects.values():
        attrs = obj.get('attributes', {})
        new_attrs = {}
        for k, v in attrs.items():
            if k == 'new_value':
                latest = _latest_real_value(v)
                if latest is None:
                    continue
            latest = _latest_real_value(v)
            if latest is None:
                continue
            new_attrs[k] = latest
        obj['attributes'] = new_attrs

def _is_real_value(v):
    return v is not None and v != '' and (not (isinstance(v, str) and v.startswith('no_')))

def _latest_real_value(v):
    if isinstance(v, list):
        real = [e for e in v if _is_real_value(e.get('value'))]
        if not real:
            return None
        real.sort(key=lambda e: e.get('time') or '')
        return real[-1].get('value')
    return v if _is_real_value(v) else None

def _load_sqlite(path: Path) -> dict:
    con = sqlite3.connect(path)
    con.row_factory = sqlite3.Row
    obj_type_to_table: dict[str, str] = {}
    for row in con.execute('SELECT ocel_type, ocel_type_map FROM object_map_type'):
        obj_type_to_table[row['ocel_type']] = row['ocel_type_map']
    evt_type_to_table: dict[str, str] = {}
    if _table_exists(con, 'event_map_type'):
        for row in con.execute('SELECT ocel_type, ocel_type_map FROM event_map_type'):
            evt_type_to_table[row['ocel_type']] = row['ocel_type_map']
    objects: dict[str, dict] = {}
    for row in con.execute('SELECT ocel_id, ocel_type FROM object'):
        objects[row['ocel_id']] = {'type': row['ocel_type'], 'attributes': {}}
    for otype, tsuffix in obj_type_to_table.items():
        tname = f'object_{tsuffix}'
        if not _table_exists(con, tname):
            continue
        cols = _table_columns(con, tname)
        attr_cols = [c for c in cols if c not in ('ocel_id', 'ocel_time', 'ocel_changed_field')]
        if not attr_cols:
            continue
        order = 'ORDER BY ocel_time' if 'ocel_time' in cols else ''
        for row in con.execute(f'SELECT * FROM {_qident(tname)} {order}'):
            oid = row['ocel_id']
            if oid not in objects:
                continue
            time = row['ocel_time'] if 'ocel_time' in cols else None
            changed_field = row['ocel_changed_field'] if 'ocel_changed_field' in cols else None
            for c in attr_cols:
                v = row[c]
                if v is None:
                    continue
                if changed_field and c != changed_field:
                    continue
                if time is not None:
                    objects[oid]['attributes'].setdefault(c, [])
                    objects[oid]['attributes'][c].append({'time': time, 'value': v})
                else:
                    objects[oid]['attributes'][c] = v
    _normalize_object_attributes(objects)
    events: dict[str, dict] = {}
    for row in con.execute('SELECT ocel_id, ocel_type FROM event'):
        events[row['ocel_id']] = {'activity': row['ocel_type'], 'timestamp': None, 'attributes': {}, 'objects': []}
    for etype, tsuffix in evt_type_to_table.items():
        tname = f'event_{tsuffix}'
        if not _table_exists(con, tname):
            continue
        cols = _table_columns(con, tname)
        attr_cols = [c for c in cols if c not in ('ocel_id', 'ocel_time', 'ocel_changed_field')]
        for row in con.execute(f'SELECT * FROM {_qident(tname)}'):
            eid = row['ocel_id']
            if eid not in events:
                continue
            if 'ocel_time' in cols and row['ocel_time'] is not None:
                events[eid]['timestamp'] = row['ocel_time']
            for c in attr_cols:
                v = row[c]
                if v is not None:
                    events[eid]['attributes'][c] = v
    if _table_exists(con, 'event_object'):
        for row in con.execute('SELECT ocel_event_id, ocel_object_id FROM event_object'):
            if row['ocel_event_id'] in events:
                events[row['ocel_event_id']]['objects'].append(row['ocel_object_id'])
    o2o = []
    if _table_exists(con, 'object_object'):
        for row in con.execute('SELECT ocel_source_id, ocel_target_id, ocel_qualifier FROM object_object'):
            o2o.append({'source': row['ocel_source_id'], 'target': row['ocel_target_id'], 'qualifier': row['ocel_qualifier'] or ''})
    con.close()
    return {'events': events, 'objects': objects, 'o2o': o2o, 'format': 'ocel2_sqlite', '_raw': str(path)}

def _qident(name: str) -> str:
    return '"' + str(name).replace('"', '""') + '"'

def _table_exists(con: sqlite3.Connection, name: str) -> bool:
    cur = con.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (name,))
    return cur.fetchone() is not None

def _table_columns(con: sqlite3.Connection, table: str) -> list[str]:
    return [row[1] for row in con.execute(f'PRAGMA table_info({_qident(table)})')]

def _export_ocel2_sqlite(log: dict, path: Path) -> None:

    def _check_unique_suffix(items: dict, kind: str) -> dict[str, str]:
        suf: dict[str, str] = {}
        seen: dict[str, str] = {}
        for t in items:
            s = t.replace(' ', '').lower()
            other = seen.get(s)
            if other is not None and other != t:
                raise ValueError(f'{kind} tsuffix collision: types {other!r} and {t!r} both map to {s!r}.  Rename one before exporting.')
            seen[s] = t
            suf[t] = s
        return suf
    obj_type_to_suffix = _check_unique_suffix({o['type']: None for o in log['objects'].values()}, 'object-type')
    evt_type_to_suffix = _check_unique_suffix({e['activity']: None for e in log['events'].values()}, 'event-type')
    if path.exists():
        path.unlink()
    con = sqlite3.connect(path)
    con.execute('PRAGMA foreign_keys = ON')
    by_type: dict = defaultdict(list)
    attr_keys_by_type: dict = defaultdict(set)
    for oid, o in log['objects'].items():
        by_type[o['type']].append((oid, o['attributes']))
        attr_keys_by_type[o['type']].update(o['attributes'].keys())
    evt_by_type: dict = defaultdict(list)
    evt_attr_keys_by_type: dict = defaultdict(set)
    for eid, e in log['events'].items():
        attrs = e.get('attributes', {})
        evt_by_type[e['activity']].append((eid, e['timestamp'], attrs))
        evt_attr_keys_by_type[e['activity']].update(attrs.keys())
    obj_type_map: list[tuple[str, str]] = [(otype, obj_type_to_suffix[otype]) for otype in by_type]
    evt_type_map: list[tuple[str, str]] = [(etype, evt_type_to_suffix[etype]) for etype in evt_by_type]
    con.execute('CREATE TABLE object_map_type (    ocel_type TEXT NOT NULL PRIMARY KEY,    ocel_type_map TEXT NOT NULL)')
    con.executemany('INSERT INTO object_map_type VALUES (?,?)', obj_type_map)
    con.execute('CREATE TABLE event_map_type (    ocel_type TEXT NOT NULL PRIMARY KEY,    ocel_type_map TEXT NOT NULL)')
    con.executemany('INSERT INTO event_map_type VALUES (?,?)', evt_type_map)
    con.execute('CREATE TABLE object (    ocel_id TEXT NOT NULL PRIMARY KEY,    ocel_type TEXT NOT NULL,    FOREIGN KEY (ocel_type) REFERENCES object_map_type(ocel_type))')
    con.executemany('INSERT INTO object VALUES (?,?)', [(oid, o['type']) for oid, o in log['objects'].items()])
    con.execute('CREATE TABLE event (    ocel_id TEXT NOT NULL PRIMARY KEY,    ocel_type TEXT NOT NULL,    FOREIGN KEY (ocel_type) REFERENCES event_map_type(ocel_type))')
    con.executemany('INSERT INTO event VALUES (?,?)', [(eid, e['activity']) for eid, e in log['events'].items()])
    for otype, rows in by_type.items():
        tsuffix = obj_type_to_suffix[otype]
        tname = f'object_{tsuffix}'
        cols = sorted(attr_keys_by_type[otype])
        col_defs = ', '.join([f'"{c}" TEXT' for c in cols])
        con.execute(f'''CREATE TABLE "{tname}" (    ocel_id TEXT NOT NULL,    ocel_time TEXT{(', ' + col_defs if col_defs else '')},    PRIMARY KEY (ocel_id, ocel_time),    FOREIGN KEY (ocel_id) REFERENCES object(ocel_id))''')
        for oid, attrs in rows:
            vals = [oid, None] + [str(attrs[c]) if c in attrs else None for c in cols]
            ph = ','.join(['?'] * len(vals))
            con.execute(f'INSERT INTO "{tname}" VALUES ({ph})', vals)
    for etype, rows in evt_by_type.items():
        tsuffix = evt_type_to_suffix[etype]
        tname = f'event_{tsuffix}'
        cols = sorted(evt_attr_keys_by_type[etype])
        col_defs = ', '.join([f'"{c}" TEXT' for c in cols])
        con.execute(f'''CREATE TABLE "{tname}" (    ocel_id TEXT NOT NULL PRIMARY KEY,    ocel_time TIMESTAMP{(', ' + col_defs if col_defs else '')},    FOREIGN KEY (ocel_id) REFERENCES event(ocel_id))''')
        for eid, ts, attrs in rows:
            vals = [eid, ts] + [str(attrs[c]) if c in attrs else None for c in cols]
            ph = ','.join(['?'] * len(vals))
            con.execute(f'INSERT INTO "{tname}" VALUES ({ph})', vals)
    known_objects: set[str] = set(log['objects'].keys())
    eo_kept: list[tuple[str, str, str]] = []
    eo_dropped = 0
    for eid, e in log['events'].items():
        for oid in e['objects']:
            if oid in known_objects:
                eo_kept.append((eid, oid, ''))
            else:
                eo_dropped += 1
    o2o_kept: list[tuple[str, str, str]] = []
    o2o_dropped = 0
    for r in log['o2o']:
        if r['source'] in known_objects and r['target'] in known_objects:
            o2o_kept.append((r['source'], r['target'], r['qualifier']))
        else:
            o2o_dropped += 1
    con.execute("CREATE TABLE event_object (    ocel_event_id TEXT NOT NULL,    ocel_object_id TEXT NOT NULL,    ocel_qualifier TEXT NOT NULL DEFAULT '',    PRIMARY KEY (ocel_event_id, ocel_object_id, ocel_qualifier),    FOREIGN KEY (ocel_event_id)  REFERENCES event(ocel_id),    FOREIGN KEY (ocel_object_id) REFERENCES object(ocel_id))")
    con.executemany('INSERT INTO event_object VALUES (?,?,?)', eo_kept)
    con.execute("CREATE TABLE object_object (    ocel_source_id TEXT NOT NULL,    ocel_target_id TEXT NOT NULL,    ocel_qualifier TEXT NOT NULL DEFAULT '',    PRIMARY KEY (ocel_source_id, ocel_target_id, ocel_qualifier),    FOREIGN KEY (ocel_source_id) REFERENCES object(ocel_id),    FOREIGN KEY (ocel_target_id) REFERENCES object(ocel_id))")
    con.executemany('INSERT INTO object_object VALUES (?,?,?)', o2o_kept)
    if eo_dropped or o2o_dropped:
        print(f'[export] dropped {eo_dropped} orphan event_object and {o2o_dropped} orphan object_object rows (referenced ids no longer present)')
    con.commit()
    con.close()

def list_object_types(log: dict) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for obj in log['objects'].values():
        counts[obj['type']] += 1
    return dict(counts)

def _try_float(v: Any) -> float | None:
    try:
        return float(v)
    except (ValueError, TypeError):
        return None

def _find_optimal_k_elbow(float_vals: list[float], max_k: int=10) -> int:
    try:
        from sklearn.cluster import KMeans
        import numpy as np
    except ImportError:
        return 3
    n_unique = len(set(float_vals))
    max_k = min(max_k, n_unique, 10)
    if max_k <= 1:
        return 1
    if max_k == 2:
        return 2
    X = np.array(float_vals).reshape(-1, 1)
    inertias = []
    for k in range(1, max_k + 1):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X)
        inertias.append(km.inertia_)
    inertias = np.array(inertias)
    inertias_norm = inertias / (inertias[0] + 1e-12)
    d1 = np.diff(inertias_norm)
    d2 = np.diff(d1)
    if len(d2) == 0:
        return max_k
    elbow_idx = int(np.argmax(d2)) + 2
    elbow_k = min(elbow_idx + 1, max_k)
    return max(2, elbow_k)

def _band_labels(k: int) -> list[str]:
    if k == 1:
        return ['all']
    if k == 2:
        return ['low', 'high']
    if k == 3:
        return ['low', 'medium', 'high']
    if k == 4:
        return ['low', 'medium-low', 'medium-high', 'high']
    if k == 5:
        return ['very_low', 'low', 'medium', 'high', 'very_high']
    return [f'Q{i + 1}' for i in range(k)]

def _collect_numeric_values(items: dict, type_key: str, type_value: str, attr: str) -> tuple[list[Any], list[float]]:
    raw_vals: list[Any] = []
    float_vals: list[float] = []
    for _, item in items.items():
        if item.get(type_key) != type_value:
            continue
        v = item.get('attributes', {}).get(attr)
        if v is None:
            continue
        f = _try_float(v)
        if f is None:
            continue
        raw_vals.append(v)
        float_vals.append(f)
    return (raw_vals, float_vals)

def discover_numeric_attributes(log: dict, kind: str='both', min_distinct: int=10) -> list[dict]:
    assert kind in ('object', 'event', 'both'), kind
    out: list[dict] = []

    def _scan(items: dict, type_key: str, kind_name: str) -> None:
        bag: dict[tuple[str, str], list[Any]] = defaultdict(list)
        for _, item in items.items():
            t = item.get(type_key)
            if t is None:
                continue
            for k, v in item.get('attributes', {}).items():
                bag[t, k].append(v)
        for (t, attr), vals in bag.items():
            non_null = [v for v in vals if v is not None]
            if len(non_null) < min_distinct:
                continue
            floats = [_try_float(v) for v in non_null]
            if any((f is None for f in floats)):
                continue
            n_distinct = len({f for f in floats})
            if n_distinct < min_distinct:
                continue
            out.append({'kind': kind_name, 'type': t, 'attr': attr, 'n_values': len(non_null), 'n_distinct': n_distinct, 'min': min(floats), 'max': max(floats), 'sample': sorted({f for f in floats})[:5]})
    if kind in ('object', 'both'):
        _scan(log['objects'], 'type', 'object')
    if kind in ('event', 'both'):
        _scan(log['events'], 'activity', 'event')
    out.sort(key=lambda r: -r['n_distinct'])
    return out

def _discretize_one(log: dict, kind: str, type_or_activity: str, attr: str, max_k: int, out_attr: str | None, labels: list[str] | None, k: int | None=None) -> dict | None:
    if kind == 'object':
        items, type_key = (log['objects'], 'type')
    elif kind == 'event':
        items, type_key = (log['events'], 'activity')
    else:
        raise ValueError(f"kind must be 'object' or 'event', got {kind!r}")
    raw_vals, float_vals = _collect_numeric_values(items, type_key, type_or_activity, attr)
    if not float_vals:
        return None
    try:
        from sklearn.cluster import KMeans
        import numpy as np
    except ImportError:
        raise ImportError('scikit-learn è necessario per la discretizzazione. Installalo con: pip install scikit-learn')
    n_distinct = len({f for f in float_vals})
    if n_distinct < 2:
        return None
    if k is None:
        k = _find_optimal_k_elbow(float_vals, max_k=max_k)
    k = max(2, min(k, n_distinct))
    X = np.array(float_vals).reshape(-1, 1)
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    cluster_ids = km.fit_predict(X)
    centroids = km.cluster_centers_.flatten()
    sort_order = np.argsort(centroids)
    rank_of = {int(cid): rank for rank, cid in enumerate(sort_order)}
    ranks = [rank_of[int(c)] for c in cluster_ids]
    label_strs = labels if labels is not None else _band_labels(k)
    if len(label_strs) != k:
        raise ValueError(f'labels has {len(label_strs)} entries, expected k={k}')
    target_attr = out_attr if out_attr is not None else f'{attr}_band'
    band_stats: list[dict] = []
    for rank in range(k):
        members = [v for v, r in zip(float_vals, ranks) if r == rank]
        band_stats.append({'label': label_strs[rank], 'min': min(members) if members else None, 'max': max(members) if members else None, 'centroid': float(centroids[sort_order[rank]]), 'count': len(members)})
    val_to_label: dict[float, str] = {}
    for fv, r in zip(float_vals, ranks):
        val_to_label[fv] = label_strs[r]
    written = 0
    for _, item in items.items():
        if item.get(type_key) != type_or_activity:
            continue
        v = item.get('attributes', {}).get(attr)
        if v is None:
            continue
        f = _try_float(v)
        if f is None:
            continue
        item['attributes'][target_attr] = val_to_label.get(f, label_strs[0])
        written += 1
    print(f"[discretize] {kind}.{type_or_activity}.{attr} → '{target_attr}'  K={k} (elbow), {written} values labelled")
    for b in band_stats:
        rng = f"≈{b['min']:.4g}" if b['min'] == b['max'] else f"[{b['min']:.4g}, {b['max']:.4g}]"
        print(f"            {b['label']:>14s}: {rng}  centroid={b['centroid']:.4g}  n={b['count']}")
    return {'kind': kind, 'type': type_or_activity, 'attr': attr, 'out_attr': target_attr, 'k': k, 'n_clustered': written, 'bands': band_stats}

def discretize_object_attribute(log: dict, object_type: str, attr: str, max_k: int=8, out_attr: str | None=None, labels: list[str] | None=None, k: int | None=None) -> dict | None:
    return _discretize_one(log, 'object', object_type, attr, max_k, out_attr, labels, k)

def discretize_event_attribute(log: dict, activity: str, attr: str, max_k: int=8, out_attr: str | None=None, labels: list[str] | None=None, k: int | None=None) -> dict | None:
    return _discretize_one(log, 'event', activity, attr, max_k, out_attr, labels, k)

def discretize_all_numeric_attributes(log: dict, min_distinct: int=10, max_k: int=8) -> list[dict]:
    candidates = discover_numeric_attributes(log, kind='both', min_distinct=min_distinct)
    print(f'[discretize] {len(candidates)} candidate attribute(s) (>= {min_distinct} distinct numeric values)')
    summaries: list[dict] = []
    for c in candidates:
        s = _discretize_one(log, c['kind'], c['type'], c['attr'], max_k=max_k, out_attr=None, labels=None)
        if s is not None:
            summaries.append(s)
    return summaries

def cardinality_of(obj_type, relation):
    left_type, right_type, left_card, right_card = relation
    if obj_type == left_type:
        return (right_type, right_card)
    if obj_type == right_type:
        return (left_type, left_card)
    return (None, None)

def build_absorption_map(absorb_candidates, keep_types):
    absorption = {}

    def resolve(t, visited=None):
        if visited is None:
            visited = set()
        if t in keep_types:
            return t
        if t in visited:
            return None
        visited.add(t)
        candidates = absorb_candidates.get(t, [])
        for c in candidates:
            if c in keep_types:
                return c
        for c in candidates:
            target = resolve(c, visited)
            if target is not None:
                return target
        return None
    for t in absorb_candidates:
        if t not in keep_types:
            absorption[t] = resolve(t)
    return absorption

def find_targets(start_oid, log, o2o_adj, ca, absorption):
    start_type = log['objects'][start_oid]['type']
    target_type = absorption[start_type]
    visited = {start_oid}
    queue = [start_oid]
    targets = []
    while queue:
        curr = queue.pop(0)
        curr_type = log['objects'][curr]['type']
        neighbors = [b if a == curr else a for a, b in o2o_adj if a == curr or b == curr]
        for nb in neighbors:
            if nb in visited or nb not in log['objects']:
                continue
            nb_type = log['objects'][nb]['type']
            if nb_type not in ca.get(curr_type, []):
                continue
            visited.add(nb)
            if nb_type == target_type:
                targets.append(nb)
            else:
                queue.append(nb)
    return targets

def get_attributes_of_type(log, target_type):
    attrs = set()
    for obj in log['objects'].values():
        if obj['type'] == target_type:
            attrs.update(obj.get('attributes', {}).keys())
    return attrs

def fill_empty_object_attributes(log, all_attr):
    for oid, obj in log['objects'].items():
        attrs = obj.setdefault('attributes', {})
        for att in all_attr[obj['type']]:
            if att not in attrs.keys():
                attrs[att] = 'no_' + att
    return log

def is_real_value(v):
    return v is not None and v != '' and (not (isinstance(v, str) and v.startswith('no_')))

def latest_value(v):
    if isinstance(v, list):
        real_entries = [e for e in v if is_real_value(e.get('value'))]
        if not real_entries:
            return None
        real_entries.sort(key=lambda e: e.get('time') or '')
        return real_entries[-1]['value']
    if is_real_value(v):
        return v
    return None

def log_summary(log: dict, title: str='SUMMARY') -> dict:
    object_types = list_object_types(log)
    activity_types = {ev.get('activity', '') for ev in log['events'].values()}
    summary = {'num_objects': len(log['objects']), 'num_object_types': len(object_types), 'num_events': len(log['events']), 'num_activity_types': len(activity_types), 'num_o2o': len(log.get('o2o', []))}
    print()
    print('─' * 60)
    print(f'  {title}')
    print('─' * 60)
    print(f"  Oggetti totali:        {summary['num_objects']}")
    print(f"  Tipi di oggetto:       {summary['num_object_types']}")
    print(f"  Eventi totali:         {summary['num_events']}")
    print(f"  Tipi di attività:      {summary['num_activity_types']}")
    print(f"  Relazioni O2O:         {summary['num_o2o']}")
    print('─' * 60)
    return summary

def clean_activity_name(name: str) -> str:
    name = name.replace("-", " ")
    name = re.sub(r"[^A-Za-z0-9 ]+", "", name)
    name = re.sub(r"\s+", " ", name).strip()
    return name


def run_auto(input_path: str, output_path: str, remove_types: list[str]) -> None:
    log = load_ocel(input_path)
    log_summary(log, "LOG PRIMA DEL FILTRO")

    print("Tipi di oggetto presenti:")
    for object_type, count in sorted(list_object_types(log).items()):
        print(f"  {object_type}: {count} oggetti")

    input_suffix = Path(input_path).suffix.lower()
    if input_suffix in (".sqlite", ".db"):
        cardinalities, o2o = o2o_cardinalities_from_sqlite(input_path)
    else:
        cardinalities, o2o = o2o_cardinalities_from_log(log)

    compact_relations = compact_o2o_relations(cardinalities)
    relations = [
        (r["left"], r["right"], r["a_to_b_range"], r["b_to_a_range"])
        for r in compact_relations
    ]

    candidate_absorptions = defaultdict(list)
    for removed_type in remove_types:
        for relation in relations:
            target_type, target_cardinality = cardinality_of(removed_type, relation)
            if target_cardinality and "*" not in target_cardinality:
                candidate_absorptions[removed_type].append(target_type)

    keep_types = set(list_object_types(log)) - set(remove_types)
    absorption = build_absorption_map(candidate_absorptions, keep_types)

    print("\n[association_by_type]")
    if not absorption:
        print("Nessun tipo rimosso viene assorbito da un altro tipo oggetto.")
    else:
        for removed_type, target_type in sorted(absorption.items()):
            if target_type is None:
                print(f"Gli attributi di {removed_type} NON hanno un target oggetto univoco")
            else:
                print(f"Gli attributi di {removed_type} sono stati messi su {target_type}")

    attributes_by_type = {
        object_type: get_attributes_of_type(log, object_type)
        for object_type in {obj["type"] for obj in log["objects"].values()}
    }
    fill_empty_object_attributes(log, attributes_by_type)

    for source_type, target_type in absorption.items():
        for _, obj in log["objects"].items():
            if obj["type"] != target_type:
                continue
            target_attrs = obj.setdefault("attributes", {})
            defaults = {
                f"{source_type.replace(' ', '_')}_{attr_name}": f"no_{attr_name}"
                for attr_name in attributes_by_type[source_type]
            }
            for attr_name, default_value in defaults.items():
                target_attrs.setdefault(attr_name, default_value)

    for oid, obj in list(log["objects"].items()):
        object_type = obj["type"]
        object_attrs = obj.get("attributes", {})

        if object_type in absorption and object_type in remove_types:
            targets = find_targets(oid, log, o2o, candidate_absorptions, absorption)
            for target_oid in targets:
                target_attrs = log["objects"][target_oid].setdefault("attributes", {})
                for attr_name, attr_value in object_attrs.items():
                    value = latest_value(attr_value)
                    if value is None:
                        continue
                    column = f"{object_type.replace(' ', '_')}_{attr_name}"
                    target_attrs[column] = value
                target_attrs[object_type] = oid

        elif object_type in remove_types:
            for event in log["events"].values():
                if oid not in event.get("objects", []):
                    continue
                event_attrs = event.setdefault("attributes", {})
                id_column = f"num_{object_type.replace(' ', '_')}_id"
                if id_column not in event_attrs:
                    event_attrs[id_column] = oid
                elif isinstance(event_attrs[id_column], list):
                    if oid not in event_attrs[id_column]:
                        event_attrs[id_column].append(oid)
                elif event_attrs[id_column] != oid:
                    event_attrs[id_column] = [event_attrs[id_column], oid]

                for type_name, attr_names in attributes_by_type.items():
                    if type_name != object_type or not attr_names:
                        continue
                    for attr_name in attr_names:
                        column = f"{object_type.replace(' ', '_')}_{attr_name.replace(' ', '')}"
                        event_attrs.setdefault(column, object_attrs[attr_name])

    remove_oids = {
        oid
        for oid, obj in log["objects"].items()
        if obj["type"] in remove_types
    }

    for oid in remove_oids:
        del log["objects"][oid]

    for event in log["events"].values():
        event["objects"] = [
            oid for oid in event.get("objects", [])
            if oid not in remove_oids
        ]

    log["o2o"] = [
        rel for rel in log.get("o2o", [])
        if rel["source"] not in remove_oids and rel["target"] not in remove_oids
    ]

    for event in log["events"].values():
        event["activity"] = clean_activity_name(event["activity"])

    type_map = {
        object_type: safe_name(object_type)
        for object_type in {obj["type"] for obj in log["objects"].values()}
    }
    for obj in log["objects"].values():
        obj["type"] = type_map[obj["type"]]

    print("[sanitize] Object type mapping:")
    for old_type, new_type in sorted(type_map.items()):
        if old_type != new_type:
            print(f"  {old_type} -> {new_type}")

    discretize_all_numeric_attributes(log, min_distinct=10)
    log_summary(log, "LOG DOPO IL FILTRO")
    export_ocel(log, output_path)


def usage() -> None:
    print("Utilizzo:")
    print("python ocel_filter.py auto <input> <output> <tipo1> [tipo2 ...]")
    sys.exit(1)


def main() -> None:
    if len(sys.argv) < 2:
        usage()

    mode = sys.argv[1].lower()
    if mode != "auto" or len(sys.argv) < 5:
        usage()

    run_auto(sys.argv[2], sys.argv[3], sys.argv[4:])


if __name__ == "__main__":
    main()
