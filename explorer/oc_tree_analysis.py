from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple

def _try_sklearn():
    try:
        from sklearn.metrics import mean_absolute_error, r2_score
        from sklearn.model_selection import train_test_split
        from sklearn.tree import DecisionTreeRegressor, export_text
        return {'ok': True, 'DecisionTreeRegressor': DecisionTreeRegressor, 'train_test_split': train_test_split, 'export_text': export_text, 'mean_absolute_error': mean_absolute_error, 'r2_score': r2_score}
    except Exception as exn:
        return {'ok': False, 'error': f'{type(exn).__name__}: {exn}'}

def _fmt_kpi_value(v: float, unit: str) -> str:
    if unit in {'s', 'second', 'seconds'}:
        if v >= 86400:
            return f'{v / 86400:.2f} d'
        if v >= 3600:
            return f'{v / 3600:.2f} h'
        if v >= 60:
            return f'{v / 60:.2f} min'
        return f'{v:.2f} s'
    if unit:
        return f'{v:.3f} {unit}'
    return f'{v:.3f}'

def _kpi_unit(state, kpi: str) -> str:
    for k in state.bundle.kpi_specs:
        if k.get('name') == kpi:
            return k.get('unit', '') or ''
    return ''

def behavior_matrix_for_cfg(state, cfg_idx: int, kpi: Optional[str]=None, after_subsumption: bool=True) -> Tuple[List[List[int]], List[float], List[int]]:
    b = state.bundle
    if kpi is None:
        kpi = b.primary_kpi
    if after_subsumption:
        pairs = state.behaviors_after_subsumption(cfg_idx)
    else:
        run = b.cfg_runs[cfg_idx]
        pairs = list(enumerate(run.patterns))
    behavior_ids = [pi for pi, _ in pairs]
    in_sets = [set(p.in_exec_idx) for _, p in pairs]
    n_exec = len(b.executions)
    kpi_values = b.kpi_values.get(kpi, [])
    if len(kpi_values) != n_exec:
        kpi_values = [0.0] * n_exec
    X = [[1 if i in s else 0 for s in in_sets] for i in range(n_exec)]
    y = [float(kpi_values[i]) for i in range(n_exec)]
    return (X, y, behavior_ids)

def _walk_tree_leaves(tree, feature_names: List[str]) -> List[Dict[str, Any]]:
    t = tree.tree_
    out: List[Dict[str, Any]] = []

    def _recurse(node: int, path: List[Dict[str, Any]]) -> None:
        if t.feature[node] == -2:
            out.append({'path': list(path), 'value': float(t.value[node][0][0]), 'n_samples': int(t.n_node_samples[node])})
            return
        feat_idx = int(t.feature[node])
        feat_name = feature_names[feat_idx] if 0 <= feat_idx < len(feature_names) else f'f{feat_idx}'
        thr = float(t.threshold[node])
        _recurse(int(t.children_left[node]), path + [{'feature': feat_name, 'value': 0}])
        _recurse(int(t.children_right[node]), path + [{'feature': feat_name, 'value': 1}])
    _recurse(0, [])
    return out

def fit_kpi_tree(X: List[List[int]], y: List[float], behavior_ids: List[int], max_depth: int=3, min_samples_leaf: int=20, test_size: float=0.25, seed: int=42) -> Dict[str, Any]:
    if not behavior_ids:
        return {'status': 'no_patterns', 'n_features': 0, 'n_samples': len(y)}
    if len(y) < max(2, min_samples_leaf):
        return {'status': 'too_few_samples', 'n_features': len(behavior_ids), 'n_samples': len(y)}
    if len(set(y)) <= 1:
        return {'status': 'constant_y', 'n_features': len(behavior_ids), 'n_samples': len(y), 'constant_value': float(y[0]) if y else 0.0}
    sk = _try_sklearn()
    if not sk['ok']:
        return {'status': 'sklearn_missing', 'error': sk['error'], 'n_features': len(behavior_ids), 'n_samples': len(y)}
    feature_names = [f'B{pi}' for pi in behavior_ids]
    try:
        tree = sk['DecisionTreeRegressor'](max_depth=max_depth, min_samples_leaf=min_samples_leaf, random_state=seed)
        tree.fit(X, y)
        pred = tree.predict(X)
        mae = float(sk['mean_absolute_error'](y, pred))
        r2 = float(sk['r2_score'](y, pred))
        text = sk['export_text'](tree, feature_names=feature_names)
        leaves = _walk_tree_leaves(tree, feature_names)
        used = sorted({lf['feature'] for lf in leaves for lf in lf.get('path', []) if isinstance(lf, dict) and 'feature' in lf})
        used = sorted({step['feature'] for leaf in leaves for step in leaf['path']})
        used_pids = [int(name.lstrip('P')) for name in used if name.startswith('P') and name[1:].isdigit()]
        leaf_vals = [lf['value'] for lf in leaves]
        max_gap = max(leaf_vals) - min(leaf_vals) if leaf_vals else 0.0
        return {'status': 'ok', 'n_features': len(behavior_ids), 'n_samples': len(y), 'n_train': len(X), 'n_test': 0, 'fit_mode': 'descriptive_full_population', 'r2': r2, 'mae': mae, 'tree_text': text, 'leaves': leaves, 'leaf_values': leaf_vals, 'max_leaf_gap': float(max_gap), 'used_features': used, 'used_behaviors': used_pids, 'n_used_behaviors': len(used_pids), 'max_depth_used': int(tree.get_depth())}
    except Exception as exn:
        return {'status': 'sklearn_failure', 'error': f'{type(exn).__name__}: {exn}', 'n_features': len(behavior_ids), 'n_samples': len(y)}

def _classify(A: set, B: set, U: set) -> str:
    if not A or not B:
        return 'trivial'
    if A == U - B:
        return 'complement'
    if A.isdisjoint(B):
        return 'mutually_exclusive'
    if A == B:
        return 'equivalent'
    if A <= B:
        return 'implies_a_to_b'
    if B <= A:
        return 'implies_b_to_a'
    return 'overlap'

def behavior_relation(state, cfg_idx: int, p_a: int, p_b: int) -> Dict[str, Any]:
    b = state.bundle
    run = b.cfg_runs[cfg_idx]
    if not 0 <= p_a < len(run.patterns) or not 0 <= p_b < len(run.patterns):
        return {'status': 'out_of_range'}
    A = set(run.patterns[p_a].in_exec_idx)
    B = set(run.patterns[p_b].in_exec_idx)
    U = set(range(len(b.executions)))
    rel = _classify(A, B, U)
    return {'status': 'ok', 'cfg_idx': cfg_idx, 'p_a': p_a, 'p_b': p_b, 'n_executions': len(U), 'n_a': len(A), 'n_b': len(B), 'n_both': len(A & B), 'n_neither': len(U - (A | B)), 'n_a_only': len(A - B), 'n_b_only': len(B - A), 'relation': rel}

def find_complement_pairs(state, cfg_idx: int, kpi: Optional[str]=None, after_subsumption: bool=True) -> List[Dict[str, Any]]:
    b = state.bundle
    if kpi is None:
        kpi = b.primary_kpi
    unit = _kpi_unit(state, kpi)
    kpi_vals = b.kpi_values.get(kpi, [])
    n_exec = len(b.executions)
    U = set(range(n_exec))
    if after_subsumption:
        pairs = state.behaviors_after_subsumption(cfg_idx)
    else:
        pairs = list(enumerate(b.cfg_runs[cfg_idx].patterns))
    in_sets = [(pi, set(p.in_exec_idx)) for pi, p in pairs]
    out: List[Dict[str, Any]] = []
    for i in range(len(in_sets)):
        pi_a, A = in_sets[i]
        for j in range(i + 1, len(in_sets)):
            pi_b, B = in_sets[j]
            if A == U - B and A and B:
                a_vals = [kpi_vals[k] for k in A if k < len(kpi_vals)]
                b_vals = [kpi_vals[k] for k in B if k < len(kpi_vals)]
                mu_a = sum(a_vals) / len(a_vals) if a_vals else 0.0
                mu_b = sum(b_vals) / len(b_vals) if b_vals else 0.0
                out.append({'p_a': pi_a, 'p_b': pi_b, 'n_a': len(A), 'n_b': len(B), 'mu_a': float(mu_a), 'mu_b': float(mu_b), 'delta': float(mu_a - mu_b), 'abs_delta': float(abs(mu_a - mu_b)), 'delta_fmt': _fmt_kpi_value(abs(mu_a - mu_b), unit)})
    out.sort(key=lambda r: r['abs_delta'], reverse=True)
    return out

def behavior_details(state, cfg_idx: int, behavior_ids: List[int], kpi: Optional[str]=None) -> List[Dict[str, Any]]:
    b = state.bundle
    if kpi is None:
        kpi = b.primary_kpi
    unit = _kpi_unit(state, kpi)
    run = b.cfg_runs[cfg_idx]
    out: List[Dict[str, Any]] = []
    for pi in behavior_ids:
        if not 0 <= pi < len(run.patterns):
            continue
        pat = run.patterns[pi]
        stats = pat.kpi_stats.get(kpi, {})
        nodes = [{'id': str(n), 'label': str(d.get('label', n))} for n, d in pat.pattern.nodes(data=True)]
        edges = []
        for u, v, d in pat.pattern.edges(data=True):
            labels = d.get('labels')
            if labels is not None:
                lbl = ' · '.join(sorted((str(x) for x in labels)))
            else:
                lbl = str(d.get('label', ''))
            edges.append({'source': str(u), 'target': str(v), 'label': lbl})
        delta = float(stats.get('delta', 0.0) or 0.0)
        signed = float(stats.get('signed_delta', delta) or 0.0)
        mu_in = float(stats.get('mu_in', 0.0) or 0.0)
        mu_out = float(stats.get('mu_out', 0.0) or 0.0)
        out.append({'behavior_idx': pi, 'support': int(pat.support), 'n_in': len(pat.in_exec_idx), 'n_out': len(pat.out_exec_idx), 'size_nodes': int(pat.size_nodes), 'size_edges': int(pat.size_edges), 'mu_in': mu_in, 'mu_out': mu_out, 'mu_in_fmt': _fmt_kpi_value(mu_in, unit), 'mu_out_fmt': _fmt_kpi_value(mu_out, unit), 'delta': delta, 'signed_delta': signed, 'delta_fmt': _fmt_kpi_value(abs(delta), unit), 'graph': {'nodes': nodes, 'edges': edges}})
    return out

def summarize_cfg_tree(state, cfg_idx: int, kpi: Optional[str]=None, max_depth: int=3, min_samples_leaf: int=20) -> Dict[str, Any]:
    b = state.bundle
    if kpi is None:
        kpi = b.primary_kpi
    unit = _kpi_unit(state, kpi)
    if not 0 <= cfg_idx < len(b.cfg_runs):
        return {'status': 'out_of_range'}
    run = b.cfg_runs[cfg_idx]
    cfg_str = state.lattice[cfg_idx].cfg_str if state.lattice else ''
    X, y, beh_ids = behavior_matrix_for_cfg(state, cfg_idx, kpi=kpi)
    tree = fit_kpi_tree(X, y, beh_ids, max_depth=max_depth, min_samples_leaf=min_samples_leaf)
    used_behaviors = tree.get('used_behaviors', [])
    pat_info = behavior_details(state, cfg_idx, used_behaviors, kpi=kpi)
    complements = find_complement_pairs(state, cfg_idx, kpi=kpi)
    max_gap = float(tree.get('max_leaf_gap', 0.0) or 0.0)
    return {'status': tree['status'], 'cfg_idx': cfg_idx, 'cfg': list(run.cfg), 'cfg_str': cfg_str, 'K': run.K, 'n_abstractions': len(beh_ids), 'kpi': kpi, 'kpi_unit': unit, 'tree': tree, 'behavior_details': pat_info, 'complement_pairs': complements, 'n_complement_pairs': len(complements), 'max_leaf_gap': max_gap, 'max_leaf_gap_fmt': _fmt_kpi_value(max_gap, unit)}

def _compute_score(gap: float, r2: float, n_pairs: int, gap_max: float, pairs_max: int) -> float:
    g_norm = gap / gap_max if gap_max > 0 else 0.0
    p_norm = n_pairs / pairs_max if pairs_max > 0 else 0.0
    r2_pos = max(0.0, min(1.0, r2))
    return 0.6 * g_norm + 0.3 * r2_pos + 0.1 * p_norm

def analyze_all_cfgs(state, kpi: Optional[str]=None, max_depth: int=6, min_samples_leaf: int=20) -> List[Dict[str, Any]]:
    b = state.bundle
    if kpi is None:
        kpi = b.primary_kpi
    unit = _kpi_unit(state, kpi)
    rows: List[Dict[str, Any]] = []
    for ci, run in enumerate(b.cfg_runs):
        if not run.is_interesting:
            continue
        kept = state.behaviors_after_subsumption(ci)
        n_kept = len(kept) if kept is not None else len(run.patterns)
        if n_kept < 1:
            continue
        X, y, beh_ids = behavior_matrix_for_cfg(state, ci, kpi=kpi)
        tree = fit_kpi_tree(X, y, beh_ids, max_depth=max_depth, min_samples_leaf=min_samples_leaf)
        complements = find_complement_pairs(state, ci, kpi=kpi)
        cfg_str = state.lattice[ci].cfg_str if state.lattice else ''
        gap = float(tree.get('max_leaf_gap', 0.0) or 0.0)
        r2 = float(tree.get('r2', 0.0) or 0.0)
        mae = float(tree.get('mae', 0.0) or 0.0)
        rows.append({'cfg_idx': ci, 'cfg': list(run.cfg), 'cfg_str': cfg_str, 'K': run.K, 'n_abstractions': n_kept, 'tree_status': tree.get('status', 'unknown'), 'tree_r2': r2, 'tree_mae': mae, 'max_leaf_gap': gap, 'max_leaf_gap_fmt': _fmt_kpi_value(gap, unit), 'used_behaviors': tree.get('used_behaviors', []), 'n_used_behaviors': tree.get('n_used_behaviors', 0), 'n_complement_pairs': len(complements)})
    if not rows:
        return rows
    gap_max = max((r['max_leaf_gap'] for r in rows), default=0.0)
    pairs_max = max((r['n_complement_pairs'] for r in rows), default=0)
    for r in rows:
        r['score'] = _compute_score(gap=r['max_leaf_gap'], r2=r['tree_r2'], n_pairs=r['n_complement_pairs'], gap_max=gap_max, pairs_max=pairs_max)
    rows.sort(key=lambda r: r['score'], reverse=True)
    return rows

def compute_leaf_membership(state, cfg_idx: int, kpi: Optional[str]=None, max_depth: int=3, min_samples_leaf: int=20, seed: int=42) -> Dict[str, Any]:
    if kpi is None:
        kpi = state.bundle.primary_kpi
    X, y, behavior_ids = behavior_matrix_for_cfg(state, cfg_idx, kpi=kpi)
    if not behavior_ids:
        return {'status': 'no_patterns', 'cfg_idx': cfg_idx, 'kpi': kpi, 'leaves': []}
    if len(y) < max(2, min_samples_leaf):
        return {'status': 'too_few_samples', 'cfg_idx': cfg_idx, 'kpi': kpi, 'n_samples': len(y), 'leaves': []}
    if len(set(y)) <= 1:
        return {'status': 'constant_y', 'cfg_idx': cfg_idx, 'kpi': kpi, 'n_samples': len(y), 'leaves': []}
    sk = _try_sklearn()
    if not sk['ok']:
        return {'status': 'sklearn_missing', 'error': sk['error'], 'cfg_idx': cfg_idx, 'kpi': kpi, 'leaves': []}
    feature_names = [f'B{pi}' for pi in behavior_ids]

    def _make_human_label(leaf_id: int, path_behaviors: List[Dict[str, Any]]) -> str:
        if not path_behaviors:
            return f'L{leaf_id} = all'
        parts: List[str] = []
        for step in path_behaviors:
            pi = step['behavior_idx']
            if step.get('presence'):
                parts.append(f'B{pi}')
            else:
                parts.append(f'¬P{pi}')
        return f'L{leaf_id} = ' + ' ∧ '.join(parts)
    try:
        tree = sk['DecisionTreeRegressor'](max_depth=max_depth, min_samples_leaf=min_samples_leaf, random_state=seed)
        tree.fit(X, y)
        leaf_node_ids = tree.apply(X)
        leaf_to_exec: Dict[int, List[int]] = {}
        for i, lid in enumerate(leaf_node_ids):
            leaf_to_exec.setdefault(int(lid), []).append(i)
        t = tree.tree_
        paths_by_node: Dict[int, List[Dict[str, Any]]] = {}
        path_behaviors_by_node: Dict[int, List[Dict[str, Any]]] = {}

        def _recurse(node: int, path: List[Dict[str, Any]], path_behaviors: List[Dict[str, Any]]) -> None:
            if t.feature[node] == -2:
                paths_by_node[int(node)] = list(path)
                path_behaviors_by_node[int(node)] = list(path_behaviors)
                return
            feat_idx = int(t.feature[node])
            feat_name = feature_names[feat_idx] if 0 <= feat_idx < len(feature_names) else f'f{feat_idx}'
            behavior_idx = int(behavior_ids[feat_idx]) if 0 <= feat_idx < len(behavior_ids) else feat_idx
            threshold = float(t.threshold[node])
            left_old_step = {'feature': feat_name, 'value': 0}
            left_pattern_step = {'behavior_idx': behavior_idx, 'feature_idx': feat_idx, 'feature': feat_name, 'threshold': threshold, 'operator': '<=', 'value': 0, 'presence': False, 'human_condition': f'B{behavior_idx} absent', 'logic': f'¬B{behavior_idx}'}
            _recurse(int(t.children_left[node]), path + [left_old_step], path_behaviors + [left_pattern_step])
            right_old_step = {'feature': feat_name, 'value': 1}
            right_pattern_step = {'behavior_idx': behavior_idx, 'feature_idx': feat_idx, 'feature': feat_name, 'threshold': threshold, 'operator': '>', 'value': 1, 'presence': True, 'human_condition': f'B{behavior_idx} present', 'logic': f'B{behavior_idx}'}
            _recurse(int(t.children_right[node]), path + [right_old_step], path_behaviors + [right_pattern_step])
        _recurse(0, [], [])
        leaves: List[Dict[str, Any]] = []
        for nid, exec_idx in leaf_to_exec.items():
            nid = int(nid)
            path = paths_by_node.get(nid, [])
            path_behaviors = path_behaviors_by_node.get(nid, [])
            human_label = _make_human_label(nid, path_behaviors)
            leaves.append({'leaf_id': nid, 'path': path, 'path_behaviors': path_behaviors, 'human_label': human_label, 'value': float(t.value[nid][0][0]), 'n_samples': len(exec_idx), 'exec_idx': list(exec_idx)})
        leaves.sort(key=lambda lf: -lf['n_samples'])
        return {'status': 'ok', 'cfg_idx': cfg_idx, 'kpi': kpi, 'n_features': len(behavior_ids), 'n_samples': len(y), 'behavior_ids': list(behavior_ids), 'feature_names': feature_names, 'tree_text': sk['export_text'](tree, feature_names=feature_names), 'leaves': leaves}
    except Exception as exn:
        return {'status': 'sklearn_failure', 'error': f'{type(exn).__name__}: {exn}', 'cfg_idx': cfg_idx, 'kpi': kpi, 'leaves': []}

def cross_leaf_distribution(state, source_cfg_idx: int, source_leaf_id: int, kpi: Optional[str]=None, target_cfgs: Optional[List[int]]=None, top_n: Optional[int]=None, max_depth: int=6, min_samples_leaf: int=20, cache: Optional[Dict[Any, Dict[str, Any]]]=None) -> Dict[str, Any]:
    import math
    b = state.bundle
    if kpi is None:
        kpi = b.primary_kpi
    unit = _kpi_unit(state, kpi)
    cache = cache if cache is not None else {}

    def _membership(ci: int) -> Dict[str, Any]:
        key = (ci, kpi)
        cached = cache.get(key)
        if cached is not None:
            return cached
        m = compute_leaf_membership(state, ci, kpi=kpi, max_depth=max_depth, min_samples_leaf=min_samples_leaf)
        cache[key] = m
        return m
    src = _membership(source_cfg_idx)
    if src['status'] != 'ok':
        return {'status': src['status'], 'targets': [], 'source_cfg_idx': source_cfg_idx, 'source_leaf_id': source_leaf_id, 'kpi': kpi, 'kpi_unit': unit}
    src_leaf = next((lf for lf in src['leaves'] if int(lf['leaf_id']) == int(source_leaf_id)), None)
    if src_leaf is None:
        return {'status': 'no_source_leaf', 'targets': [], 'source_cfg_idx': source_cfg_idx, 'source_leaf_id': source_leaf_id, 'kpi': kpi, 'kpi_unit': unit}
    src_exec_set = set(src_leaf['exec_idx'])
    n_src = len(src_exec_set)
    source_human_label = src_leaf.get('human_label', f'L{source_leaf_id}')
    source_path_behaviors = src_leaf.get('path_behaviors', [])
    if target_cfgs is None:
        target_cfgs = [ci for ci, run in enumerate(b.cfg_runs) if ci != source_cfg_idx and run.is_interesting]
    kpi_vals = b.kpi_values.get(kpi, [])
    out: List[Dict[str, Any]] = []
    for tci in target_cfgs:
        tgt = _membership(tci)
        if tgt['status'] != 'ok':
            continue
        tgt_leaves: List[Dict[str, Any]] = []
        for lf in tgt['leaves']:
            tgt_set = set(lf['exec_idx'])
            overlap = src_exec_set & tgt_set
            n_overlap = len(overlap)
            if n_overlap == 0:
                continue
            mean_kpi = sum((kpi_vals[i] for i in overlap)) / n_overlap if kpi_vals else 0.0
            frac_of_source = n_overlap / n_src if n_src else 0.0
            frac_of_target_leaf = n_overlap / len(tgt_set) if tgt_set else 0.0
            tgt_leaves.append({'leaf_id': lf['leaf_id'], 'path': lf.get('path', []), 'human_label': lf.get('human_label', f"L{lf['leaf_id']}"), 'path_behaviors': lf.get('path_behaviors', []), 'n_overlap': n_overlap, 'exec_idx': sorted(overlap), 'n_total_in_leaf': len(tgt_set), 'frac_of_source': frac_of_source, 'frac_of_target_leaf': frac_of_target_leaf, 'mean_kpi': float(mean_kpi), 'mean_kpi_fmt': _fmt_kpi_value(mean_kpi, unit)})
        if not tgt_leaves:
            continue
        tgt_leaves.sort(key=lambda r: -r['n_overlap'])
        ps = [l['frac_of_source'] for l in tgt_leaves if l['frac_of_source'] > 0]
        entropy = -sum((p * math.log2(p) for p in ps)) if ps else 0.0
        cfg_str = state.lattice[tci].cfg_str if state.lattice else ''
        out.append({'cfg_idx': tci, 'cfg_str': cfg_str, 'n_leaves': len(tgt_leaves), 'entropy': entropy, 'n_source_exec': n_src, 'n_matched': sum((l['n_overlap'] for l in tgt_leaves)), 'leaves': tgt_leaves})
    out.sort(key=lambda r: -r['entropy'])
    if top_n is not None:
        out = out[:top_n]
    return {'status': 'ok', 'source_cfg_idx': source_cfg_idx, 'source_leaf_id': source_leaf_id, 'source_n_exec': n_src, 'source_path': src_leaf.get('path', []), 'source_leaf': src_leaf, 'source_human_label': source_human_label, 'source_path_behaviors': source_path_behaviors, 'source_value': src_leaf['value'], 'source_value_fmt': _fmt_kpi_value(src_leaf['value'], unit), 'kpi': kpi, 'kpi_unit': unit, 'targets': out}
