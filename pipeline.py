"""
pipeline.py  -  Shared inference pipeline (used by api.py and compare_clf_hmm.py)
================================================================================
One place that produces all three predictions for a trip:

  1. classifier-only : argmax classifier score per GPS observation
  2. HMM-only        : original Gaussian-emission Viterbi (map_matching.py)
  3. classifier+HMM  : Viterbi whose emission = log(LightGBM probability)

and the metrics that go with them (top-1 accuracy, route F1, route continuity,
candidate miss rate).
"""
import os, math, time
import numpy as np

from map_matching import (load_nodes, load_arcs, load_track, load_route,
                          viterbi_match, compress_route, expand_route,
                          route_overlap_score, haversine, dijkstra_node,
                          clear_cache, INF)
from features import extract_features

BETA = 25.0


# ── metrics ───────────────────────────────────────────────────────────────────

def route_continuity(route, arcs):
    """Fraction of consecutive arc pairs where dst(arc_i) == src(arc_i+1)."""
    if len(route) < 2:
        return 1.0
    ok = sum(1 for i in range(len(route) - 1)
             if arcs[route[i]][1] == arcs[route[i + 1]][0])
    return ok / (len(route) - 1)


def top1_accuracy(per_obs_arcs, gt_set):
    """Share of observations whose chosen arc is in the ground-truth route."""
    if not per_obs_arcs or not gt_set:
        return None
    return sum(1 for a in per_obs_arcs if a in gt_set) / len(per_obs_arcs)


def candidate_miss_rate(obs_arc_ids, gt_set):
    """Share of observations where NO candidate arc is in the GT route."""
    if not obs_arc_ids or not gt_set:
        return None
    miss = sum(1 for c in obs_arc_ids if not any(a in gt_set for a in c))
    return miss / len(obs_arc_ids)


def _metrics(route, per_obs, arcs, gt_set, gt_list):
    m = {'route_continuity': route_continuity(route, arcs)}
    if gt_list:
        p, r, f1 = route_overlap_score(route, gt_list)
        m.update(precision=p, recall=r, macro_f1=f1,
                 top1_accuracy=top1_accuracy(per_obs, gt_set))
    return m


# ── classifier-weighted Viterbi ───────────────────────────────────────────────

def _viterbi_clf(sub_track, obs_arc_ids, obs_scores, nodes, arcs, adj):
    """Viterbi over candidate arcs with emission = log(classifier prob)."""
    n = len(sub_track)
    if n == 0 or not obs_arc_ids[0]:
        return []

    def emit(obs_i, aid):
        return math.log(max(obs_scores[obs_i].get(aid, 0.1), 1e-6))

    V = [{aid: (emit(0, aid), None) for aid in obs_arc_ids[0]}]

    for t in range(1, n):
        lon_p, lat_p, _ = sub_track[t - 1]
        lon_c, lat_c, _ = sub_track[t]
        gps_m  = haversine(lon_p, lat_p, lon_c, lat_c)
        radius = max(gps_m * 3 + 300, 1500.0)

        prev_alive = [(a, lp) for a, (lp, _) in V[t - 1].items() if lp > -1e28]
        reach = {dn: dijkstra_node(dn, nodes, adj, max_m=radius)
                 for dn in {arcs[a][1] for a, _ in prev_alive}}

        cur = {}
        for aid in obs_arc_ids[t]:
            em, src_n = emit(t, aid), arcs[aid][0]
            best_lp, best_prev = -1e30, None
            for pa, prev_lp in prev_alive:
                road_m = 0.0 if pa == aid else reach[arcs[pa][1]].get(src_n, INF)
                tr = -1e9 if road_m >= INF else -(abs(road_m - gps_m) / BETA)
                lp = prev_lp + tr + em
                if lp > best_lp:
                    best_lp, best_prev = lp, pa
            cur[aid] = (best_lp, best_prev)

        # broken chain: restart from emissions only (same rule as HMM-only)
        if not cur or max(v[0] for v in cur.values()) <= -1e28:
            cur = {aid: (emit(t, aid), None) for aid in obs_arc_ids[t]}
        V.append(cur)

    best_last = max(V[-1], key=lambda a: V[-1][a][0])
    path = [best_last]
    for t in range(n - 1, 0, -1):
        prev = V[t][path[-1]][1]
        if prev is None:
            break
        path.append(prev)
    path.reverse()
    return path


# ── main entry ────────────────────────────────────────────────────────────────

def run_trip(folder, trip_id, clf=None):
    """
    Run every method on one trip folder.  Returns a dict with routes (arc-id
    lists), metrics and the raw GPS track.  `clf` may be None (HMM-only).
    """
    clear_cache()          # Dijkstra cache is keyed by node id only -> must be
                           # reset between trips (each trip has its own graph)
    base      = os.path.join(folder, trip_id)
    nodes     = load_nodes(base + '.nodes')
    arcs, adj = load_arcs(base + '.arcs')
    track     = load_track(base + '.track')

    gt_list = load_route(base + '.route') if os.path.exists(base + '.route') else []
    gt_set  = set(gt_list)

    # HMM-only (original method)
    t0 = time.perf_counter()
    hmm_obs   = viterbi_match(track, nodes, arcs, adj, subsample=True)
    hmm_route = expand_route(compress_route(hmm_obs), nodes, arcs, adj)

    t_hmm = time.perf_counter() - t0

    out = dict(nodes=nodes, arcs=arcs, track=track,
               timing={'hmm_only_s': t_hmm},
               routes={'hmm_only': hmm_route},
               metrics={'hmm_only': _metrics(hmm_route, hmm_obs, arcs, gt_set, gt_list)},
               mode='hmm_only')

    if clf is None:
        return out

    # classifier scores (features are extracted once, candidates reused)
    t1 = time.perf_counter()
    from map_matching import subsample_track
    X, _, g, obs_arc_ids = extract_features(folder, trip_id, subsample=True)
    scores = clf.predict_proba(X)[:, 1]

    row_to_arc = [a for aids in obs_arc_ids for a in aids]
    obs_scores = [dict() for _ in obs_arc_ids]
    for row_i, (oid, aid) in enumerate(zip(g, row_to_arc)):
        obs_scores[oid][aid] = float(scores[row_i])

    # classifier-only: argmax per observation, de-duplicated (no gap filling)
    clf_obs = [max(sc, key=sc.get) for sc in obs_scores if sc]
    clf_route = compress_route(clf_obs)

    # classifier + HMM
    sub_track = subsample_track(track)
    ch_obs    = _viterbi_clf(sub_track, obs_arc_ids, obs_scores, nodes, arcs, adj)
    ch_route  = expand_route(compress_route(ch_obs), nodes, arcs, adj)

    out['timing']['classifier_hmm_s'] = time.perf_counter() - t1
    out['routes'].update(classifier_only=clf_route, classifier_hmm=ch_route)
    out['metrics'].update(
        classifier_only=_metrics(clf_route, clf_obs, arcs, gt_set, gt_list),
        classifier_hmm=_metrics(ch_route, ch_obs, arcs, gt_set, gt_list))
    out['candidate_miss_rate'] = candidate_miss_rate(obs_arc_ids, gt_set)
    out['mode'] = 'classifier+hmm'
    return out
