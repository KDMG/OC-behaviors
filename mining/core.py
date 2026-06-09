
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, FrozenSet, List, Set, Tuple

import networkx as nx
from pm4py.objects.ocel.obj import OCEL

EV_ID     = "ocel:eid"
OBJ_ID    = "ocel:oid"
OBJ_TYPE  = "ocel:type"
ACTIVITY  = "ocel:activity"
TIMESTAMP = "ocel:timestamp"

Edge = Tuple[str, str, str]

@dataclass
class ProcessExecution:
    objects:    FrozenSet[str]
    events:     Set[str]
    edges:      List[Edge]
    event_info: Dict[str, Dict]


@dataclass
class ObjectTypeHierarchy:
    object_type: str
    levels:      List[Set]
    mappings:    List[Callable]
    cover_up:    Dict[int, List[int]] = field(default_factory=dict)
    cover_dn:    Dict[int, List[int]] = field(default_factory=dict)

    def apply(self, o: Any, level: int) -> Any:
        if level == 0:
            return o
        return self.mappings[level - 1](o)

    @property
    def k(self) -> int:
        return len(self.levels) - 1

    def covers(self, lev: int) -> List[int]:
        if self.cover_up:
            return list(self.cover_up.get(lev, []))
        k = self.k
        if lev == k:
            return []
        if lev == 0:
            return [1] if k <= 1 else list(range(1, k))
        return [k]

    def covered_by(self, lev: int) -> List[int]:
        if self.cover_dn:
            return list(self.cover_dn.get(lev, []))
        k = self.k
        if lev == 0:
            return []
        if lev == k:
            return [0] if k <= 1 else list(range(1, k))
        return [0]

    def height(self, lev: int) -> int:
        if self.cover_up:
            if lev == 0:
                return 0
            h: Dict[int, int] = {0: 0}

            def _h(v: int) -> int:
                if v in h:
                    return h[v]
                parents = self.cover_dn.get(v, [])
                if not parents:
                    h[v] = 0
                    return 0
                h[v] = 1 + max(_h(p) for p in parents)
                return h[v]

            return _h(lev)
        k = self.k
        if k <= 1:
            return lev
        if lev == 0:
            return 0
        if lev == k:
            return 2
        return 1


class LevelConfiguration:
    def __init__(
        self,
        config: Dict[str, int],
        hierarchies: Dict[str, ObjectTypeHierarchy],
        obj_types: Dict[str, str],
    ) -> None:
        self.config = config
        self.hierarchies = hierarchies
        self.obj_types = obj_types

    def relabel(self, o: str) -> Any:
        tau = self.obj_types[o]
        lv  = self.config.get(tau, 0)
        return o if lv == 0 else self.hierarchies[tau].apply(o, lv)


@dataclass
class BehaviorGraph:
    abstract_nodes: Set[int]
    abstract_edges: Set[int]
    node_activity:  Dict[int, str]
    node_objects: Dict[int, FrozenSet]
    edge_label: Dict[int, Any]
    edge_kappa: Dict[int, Tuple[str, str]]
    edge_endpoints: Dict[int, Tuple[int, int]]
    alpha: Dict[str, int]
    beta: Dict[int, int]

@dataclass(frozen=True)
class BehaviorFootprint:
    n_nodes: int
    n_edges: int
    act_multiset: FrozenSet
    obj_multiset: FrozenSet
    edge_multiset: FrozenSet
    degree_seq: Tuple


def compute_footprint(beh: BehaviorGraph) -> BehaviorFootprint:
    if not beh.abstract_nodes:
        return BehaviorFootprint(0, 0, frozenset(), frozenset(), frozenset(), ())

    from collections import Counter
    act_cnt = Counter(beh.node_activity.values())
    act_ms  = frozenset(act_cnt.items())

    obj_pairs = frozenset(
        pair
        for pairs in beh.node_objects.values()
        for pair in pairs
    )

    edge_ms = frozenset(
        (str(beh.edge_label[ae]), beh.edge_kappa[ae][0], beh.edge_kappa[ae][1])
        for ae in beh.abstract_edges
    )

    in_deg:  Dict[int, int] = defaultdict(int)
    out_deg: Dict[int, int] = defaultdict(int)
    for ae in beh.abstract_edges:
        s, t = beh.edge_endpoints[ae]
        out_deg[s] += 1
        in_deg[t]  += 1

    deg_seq = tuple(sorted(
        (in_deg.get(n, 0), out_deg.get(n, 0))
        for n in beh.abstract_nodes
    ))

    return BehaviorFootprint(
        n_nodes = len(beh.abstract_nodes),
        n_edges = len(beh.abstract_edges),
        act_multiset = act_ms,
        obj_multiset = obj_pairs,
        edge_multiset = edge_ms,
        degree_seq = deg_seq,
    )

def _index_ocel(ocel: OCEL):
    e2o: Dict[str, Set[str]] = defaultdict(set)
    for _, row in ocel.relations.iterrows():
        e2o[row[EV_ID]].add(row[OBJ_ID])

    ts  = dict(zip(ocel.events[EV_ID], ocel.events[TIMESTAMP]))
    act = dict(zip(ocel.events[EV_ID], ocel.events[ACTIVITY]))

    o2tr = {
        oid: sorted(grp[EV_ID].tolist(), key=lambda e: ts.get(e))
        for oid, grp in ocel.relations.groupby(OBJ_ID)
    }
    einfo = {
        e: {"activity": act[e], "timestamp": ts[e], "objects": e2o[e]}
        for e in ocel.events[EV_ID]
    }
    return e2o, o2tr, einfo


def extract_process_executions(ocel: OCEL) -> List[ProcessExecution]:
    G: nx.Graph = nx.Graph()
    G.add_nodes_from(ocel.objects[OBJ_ID])
    e2o, o2tr, einfo = _index_ocel(ocel)

    for _, grp in ocel.relations.groupby(EV_ID):
        objs = grp[OBJ_ID].tolist()
        for i in range(len(objs)):
            for j in range(i + 1, len(objs)):
                G.add_edge(objs[i], objs[j])

    result: List[ProcessExecution] = []
    for comp in nx.connected_components(G):
        X = frozenset(comp)
        E_X = {e for e, obs in e2o.items() if X & obs}
        D_X: List[Edge] = []
        for o in X:
            trace = [e for e in o2tr.get(o, []) if e in E_X]
            for i in range(len(trace) - 1):
                D_X.append((trace[i], trace[i + 1], o))
        result.append(ProcessExecution(X, E_X, D_X, {e: einfo[e] for e in E_X}))
    return result

class OCELLevelGraph:
    def __init__(self, ex: ProcessExecution, lc: LevelConfiguration) -> None:
        self.ex = ex
        self.lc = lc

    @property
    def nodes(self) -> Set[str]:
        return self.ex.events

    @property
    def edges(self) -> List[Edge]:
        return self.ex.edges

    def lambda_act(self, e: str) -> str:
        return self.ex.event_info[e]["activity"]

    def lambda_C(self, edge: Edge) -> Any:
        return self.lc.relabel(edge[2])

    def lambda_obj(self, n: str) -> FrozenSet[Tuple[Any, str]]:
        cnt: Dict[Any, int] = defaultdict(int)
        for o in self.ex.event_info[n]["objects"]:
            cnt[self.lc.relabel(o)] += 1
        return frozenset((u, "1" if c == 1 else "*") for u, c in cnt.items())


def compute_behavior(lg: OCELLevelGraph) -> BehaviorGraph:
    nodes_list = list(lg.nodes)
    edges_list = lg.edges
    n = len(nodes_list)
    m = len(edges_list)

    if n == 0:
        empty: Dict = {}
        return BehaviorGraph(set(), set(), empty, empty, empty, empty,
                             empty, empty, empty)

    node_to_idx: Dict[str, int] = {v: i for i, v in enumerate(nodes_list)}
    act_intern: Dict[str, int] = {}
    obj_intern: Dict[FrozenSet, int] = {}
    act_id:  List[int] = [0] * n
    obj_id:  List[int] = [0] * n
    act_for: List[str] = [""] * n
    obj_for: List[FrozenSet] = [frozenset()] * n
    for i, v in enumerate(nodes_list):
        a = lg.lambda_act(v)
        ai = act_intern.get(a)
        if ai is None:
            ai = len(act_intern)
            act_intern[a] = ai
        act_id[i]  = ai
        act_for[i] = a
        o = lg.lambda_obj(v)
        oi = obj_intern.get(o)
        if oi is None:
            oi = len(obj_intern)
            obj_intern[o] = oi
        obj_id[i]  = oi
        obj_for[i] = o

    lbl_intern: Dict[Any, int] = {}
    edge_src: List[int] = [0] * m
    edge_tgt: List[int] = [0] * m
    edge_lbl: List[int] = [0] * m
    edge_lbl_orig: List[Any] = [None] * m
    in_e:  List[List[int]] = [[] for _ in range(n)]
    out_e: List[List[int]] = [[] for _ in range(n)]
    for k, e in enumerate(edges_list):
        s = node_to_idx[e[0]]
        t = node_to_idx[e[1]]
        l = lg.lambda_C(e)
        li = lbl_intern.get(l)
        if li is None:
            li = len(lbl_intern)
            lbl_intern[l] = li
        edge_src[k] = s
        edge_tgt[k] = t
        edge_lbl[k] = li
        edge_lbl_orig[k] = l
        out_e[s].append(k)
        in_e[t].append(k)

    part: List[int] = [0] * n
    blocks: Dict[int, List[int]] = {}
    init_key: Dict[Tuple[int, int], int] = {}
    for i in range(n):
        key = (act_id[i], obj_id[i])
        bid = init_key.get(key)
        if bid is None:
            bid = len(init_key)
            init_key[key] = bid
            blocks[bid] = []
        part[i] = bid
        blocks[bid].append(i)
    n_blocks = len(blocks)

    dirty: Set[int] = set(blocks.keys())

    edge_src_l = edge_src
    edge_tgt_l = edge_tgt
    edge_lbl_l = edge_lbl
    in_e_l = in_e
    out_e_l = out_e
    part_l = part

    while dirty:
        perturbed_in: Dict[int, Set[int]] = {}
        for b in dirty:
            for j in blocks[b]:
                for k in in_e_l[j]:
                    src = edge_src_l[k]
                    src_blk = part_l[src]
                    s = perturbed_in.get(src_blk)
                    if s is None:
                        s = set()
                        perturbed_in[src_blk] = s
                    s.add(src)
                for k in out_e_l[j]:
                    tgt = edge_tgt_l[k]
                    tgt_blk = part_l[tgt]
                    s = perturbed_in.get(tgt_blk)
                    if s is None:
                        s = set()
                        perturbed_in[tgt_blk] = s
                    s.add(tgt)

        plan: List[Tuple[int, Dict[Tuple[FrozenSet, FrozenSet], List[int]]]] = []
        for b, perturbed in perturbed_in.items():
            members = blocks[b]
            if len(members) <= 1:
                continue

            sig_groups: Dict[Tuple[FrozenSet, FrozenSet], List[int]] = {}

            n_perturbed = len(perturbed)
            if n_perturbed < len(members):
                rep = -1
                for i in members:
                    if i not in perturbed:
                        rep = i
                        break
                if rep != -1:
                    ins_rep = frozenset(
                        (edge_lbl_l[k], part_l[edge_src_l[k]])
                        for k in in_e_l[rep]
                    )
                    outs_rep = frozenset(
                        (edge_lbl_l[k], part_l[edge_tgt_l[k]])
                        for k in out_e_l[rep]
                    )
                    stable_list = [i for i in members if i not in perturbed]
                    sig_groups[(ins_rep, outs_rep)] = stable_list

            for i in perturbed:
                ins = frozenset(
                    (edge_lbl_l[k], part_l[edge_src_l[k]])
                    for k in in_e_l[i]
                )
                outs = frozenset(
                    (edge_lbl_l[k], part_l[edge_tgt_l[k]])
                    for k in out_e_l[i]
                )
                sig = (ins, outs)
                lst = sig_groups.get(sig)
                if lst is None:
                    sig_groups[sig] = [i]
                else:
                    lst.append(i)

            if len(sig_groups) <= 1:
                continue

            plan.append((b, sig_groups))

        new_dirty: Set[int] = set()
        for b, sig_groups in plan:
            sub_lists = sorted(sig_groups.values(), key=len, reverse=True)
            blocks[b] = sub_lists[0]
            for ms in sub_lists[1:]:
                nb = n_blocks
                n_blocks += 1
                for i in ms:
                    part_l[i] = nb
                blocks[nb] = ms
                new_dirty.add(nb)

        dirty = new_dirty

    alpha: Dict[str, int] = {}
    nact:  Dict[int, str] = {}
    nobj:  Dict[int, FrozenSet] = {}
    for i, v in enumerate(nodes_list):
        b = part[i]
        alpha[v] = b
        if b not in nact:
            nact[b] = act_for[i]
            nobj[b] = obj_for[i]

    ak2i: Dict[Tuple[int, int, Any], int] = {}
    beta: Dict[int, int] = {}
    for k in range(m):
        key = (part[edge_src[k]], part[edge_tgt[k]], edge_lbl_orig[k])
        ai = ak2i.get(key)
        if ai is None:
            ai = len(ak2i)
            ak2i[key] = ai
        beta[k] = ai

    apre: Dict[int, List[int]] = defaultdict(list)
    for k in range(m):
        apre[beta[k]].append(k)

    el: Dict[int, Any] = {}
    ek: Dict[int, Tuple[str, str]] = {}
    ee: Dict[int, Tuple[int, int]] = {}
    for (sc, tc, lbl), ai in ak2i.items():
        src_first_tgt: Dict[int, int] = {}
        tgt_first_src: Dict[int, int] = {}

        ks_many = False
        kt_many = False

        for k in apre[ai]:
            s = edge_src[k]
            t = edge_tgt[k]

            old_t = src_first_tgt.get(s)
            if old_t is None:
                src_first_tgt[s] = t
            elif old_t != t:
                ks_many = True

            old_s = tgt_first_src.get(t)
            if old_s is None:
                tgt_first_src[t] = s
            elif old_s != s:
                kt_many = True

            if ks_many and kt_many:
                break

        el[ai] = lbl
        ek[ai] = ("*" if ks_many else "1", "*" if kt_many else "1")
        ee[ai] = (sc, tc)

    return BehaviorGraph(
        set(part), set(ak2i.values()),
        nact, nobj, el, ek, ee, alpha, beta,
    )



def _beh_to_nx(beh: BehaviorGraph) -> nx.DiGraph:
    G: nx.DiGraph = nx.DiGraph()
    for n in beh.abstract_nodes:
        G.add_node(n,
                   act=beh.node_activity[n],
                   obj=frozenset(beh.node_objects[n]))

    by_pair: Dict[Tuple[int, int], List] = defaultdict(list)
    for ae in beh.abstract_edges:
        s, t = beh.edge_endpoints[ae]
        by_pair[(s, t)].append((str(beh.edge_label[ae]), beh.edge_kappa[ae]))

    for (s, t), attrs in by_pair.items():
        G.add_edge(s, t, edges=tuple(sorted(attrs)))
    return G


def _iso_node_match(d1: Dict, d2: Dict) -> bool:
    return d1["act"] == d2["act"] and d1["obj"] == d2["obj"]


def _iso_edge_match(d1: Dict, d2: Dict) -> bool:
    return d1["edges"] == d2["edges"]


def _are_isomorphic_nx(g1: "nx.DiGraph", g2: "nx.DiGraph") -> bool:

    if g1.number_of_nodes() != g2.number_of_nodes():
        return False
    if g1.number_of_edges() != g2.number_of_edges():
        return False
    return nx.is_isomorphic(g1, g2,
                            node_match=_iso_node_match,
                            edge_match=_iso_edge_match)


def are_isomorphic(b1: BehaviorGraph, b2: BehaviorGraph) -> bool:
    if len(b1.abstract_nodes) != len(b2.abstract_nodes):
        return False
    if len(b1.abstract_edges) != len(b2.abstract_edges):
        return False
    if compute_footprint(b1) != compute_footprint(b2):
        return False
    return _are_isomorphic_nx(_beh_to_nx(b1), _beh_to_nx(b2))


def compute_wl_hash(beh: BehaviorGraph, max_iter: int = 4) -> int:
    nodes = beh.abstract_nodes
    if not nodes:
        return hash(())

    in_lbl:  Dict[int, List[Tuple[Any, int]]] = defaultdict(list)
    out_lbl: Dict[int, List[Tuple[Any, int]]] = defaultdict(list)
    for ae in beh.abstract_edges:
        s, t = beh.edge_endpoints[ae]
        lp = (str(beh.edge_label[ae]), beh.edge_kappa[ae])
        in_lbl[t].append((lp, s))
        out_lbl[s].append((lp, t))

    intern: Dict[Any, int] = {}
    color:  Dict[int, int] = {}
    for n in nodes:
        c = (beh.node_activity[n], beh.node_objects[n])
        cid = intern.get(c)
        if cid is None:
            cid = len(intern)
            intern[c] = cid
        color[n] = cid

    prev_n_classes = len(intern)
    for _ in range(max_iter):
        new_intern: Dict[Tuple, int] = {}
        new_color:  Dict[int, int] = {}
        for n in nodes:
            ins = tuple(sorted((lp, color[s]) for lp, s in in_lbl[n]))
            outs = tuple(sorted((lp, color[t]) for lp, t in out_lbl[n]))
            sig = (color[n], ins, outs)
            cid = new_intern.get(sig)
            if cid is None:
                cid = len(new_intern)
                new_intern[sig] = cid
            new_color[n] = cid
        if len(new_intern) == prev_n_classes:
            color = new_color
            break
        color = new_color
        prev_n_classes = len(new_intern)

    return hash(tuple(sorted(color.values())))


def compute_iso_classes_old(behaviors: List[BehaviorGraph]) -> List[int]:
    n = len(behaviors)
    if n == 0:
        return []

    ids: List[int] = [-1] * n
    fps = [compute_footprint(b) for b in behaviors]

    fp_groups: Dict[Any, List[int]] = defaultdict(list)
    for i in range(n):
        fp_groups[fps[i]].append(i)

    nxt = 0
    nxg_cache: Dict[int, "nx.DiGraph"] = {}

    def _nxg(i: int) -> "nx.DiGraph":
        g = nxg_cache.get(i)
        if g is None:
            g = _beh_to_nx(behaviors[i])
            nxg_cache[i] = g
        return g

    for fp_group in fp_groups.values():
        if len(fp_group) == 1:
            ids[fp_group[0]] = nxt
            nxt += 1
            continue

        wl_groups: Dict[int, List[int]] = defaultdict(list)
        for i in fp_group:
            wl_groups[compute_wl_hash(behaviors[i])].append(i)

        for wl_group in wl_groups.values():
            if len(wl_group) == 1:
                ids[wl_group[0]] = nxt
                nxt += 1
                continue

            reps: List[int] = []
            for i in wl_group:
                assigned = False
                gi = _nxg(i)
                for r in reps:
                    if _are_isomorphic_nx(gi, _nxg(r)):
                        ids[i] = ids[r]
                        assigned = True
                        break
                if not assigned:
                    ids[i] = nxt
                    reps.append(i)
                    nxt += 1

    return ids

def compute_iso_classes(behaviors: List[BehaviorGraph]) -> List[int]:
    n = len(behaviors)
    if n == 0:
        return []

    ids: List[int] = [-1] * n
    fps = [compute_footprint(b) for b in behaviors]

    fp_groups: Dict[Any, List[int]] = defaultdict(list)
    for i in range(n):
        fp_groups[fps[i]].append(i)

    nxt = 0
    nxg_cache: Dict[int, "nx.DiGraph"] = {}

    def _nxg(i: int) -> "nx.DiGraph":
        g = nxg_cache.get(i)
        if g is None:
            g = _beh_to_nx(behaviors[i])
            nxg_cache[i] = g
        return g

    for fp_group in fp_groups.values():
        reps: List[int] = []

        for i in fp_group:
            assigned = False
            gi = _nxg(i)

            for r in reps:
                if _are_isomorphic_nx(gi, _nxg(r)):
                    ids[i] = ids[r]
                    assigned = True
                    break

            if not assigned:
                ids[i] = nxt
                reps.append(i)
                nxt += 1

    return ids