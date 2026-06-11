
from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, FrozenSet, List, Optional, Set, Tuple

import networkx as nx
from networkx.algorithms.isomorphism import DiGraphMatcher

PatternLike = Any


def _node_match(a: Dict, b: Dict) -> bool:
    return str(a.get("label", "")) == str(b.get("label", ""))


def _edge_label_set(d: Dict) -> FrozenSet[str]:
    labels = d.get("labels")
    if labels is not None:
        return frozenset(str(x) for x in labels)
    lbl = d.get("label")
    return frozenset({str(lbl)}) if lbl is not None else frozenset()


def _edge_match(a: Dict, b: Dict) -> bool:
    return _edge_label_set(a) == _edge_label_set(b)


def is_subgraph_of(small: nx.DiGraph, big: nx.DiGraph) -> bool:
    if small.number_of_nodes() == 0:
        return True
    if small.number_of_nodes() > big.number_of_nodes():
        return False
    if small.number_of_edges() > big.number_of_edges():
        return False
    matcher = DiGraphMatcher(big, small,
                             node_match=_node_match,
                             edge_match=_edge_match)
    return matcher.subgraph_is_isomorphic()



def _bucket_by_support(
        patterns: List[PatternLike],
) -> Dict[FrozenSet[int], List[int]]:
    buckets: Dict[FrozenSet[int], List[int]] = defaultdict(list)
    for pi, p in enumerate(patterns):
        buckets[frozenset(p.in_exec_idx)].append(pi)
    return buckets


def compute_kept_patterns(
        patterns: List[PatternLike],
) -> Optional[Set[int]]:
    if len(patterns) <= 1:
        return None

    buckets = _bucket_by_support(patterns)
    if all(len(v) <= 1 for v in buckets.values()):
        return None

    kept: Set[int] = set()
    for _support_set, idxs in buckets.items():
        if len(idxs) == 1:
            kept.add(idxs[0])
            continue

        ordered = sorted(
            idxs,
            key=lambda pi: (patterns[pi].size_edges, patterns[pi].size_nodes),
            reverse=True,
        )
        anchors: List[int] = []
        for pi in ordered:
            small = patterns[pi].pattern
            subsumed = False
            for ai in anchors:
                big = patterns[ai].pattern
                if small.number_of_edges() > big.number_of_edges():
                    continue
                if is_subgraph_of(small, big):
                    subsumed = True
                    break
            if not subsumed:
                anchors.append(pi)

        kept.update(anchors)

    return kept
