"""
PathSnap Map-Matching - HMM Viterbi over road arcs
===================================================
.nodes  : lon lat          (0-indexed)
.arcs   : src dst          (0-indexed arc list)
.track  : lon lat ts       (GPS observations)
.route  : arc_id           (ground truth - one arc ID per line)

Speed optimisations:
  - GPS subsampling (keep ~1 point per SUBSAMPLE_M metres of movement)
  - Dijkstra cache keyed on (src_node, max_m)
  - Early -inf pruning
"""

import os, math, heapq
from collections import defaultdict

# ── constants ─────────────────────────────────────────────────────────────────
SIGMA_GPS    = 15.0    # GPS noise std-dev (metres)
BETA         = 25.0    # transition exponential scale (metres) — higher = prefer shorter routes
MAX_CAND     = 6       # candidates per GPS observation
MAX_CAND_M   = 50.0    # max snap dist (metres)
DIJK_MAX     = 1500.0  # Dijkstra radius (metres)
SUBSAMPLE_M  = 25.0    # keep GPS points ≥ this many metres apart — higher = fewer obs, faster
INF          = float('inf')


# ── geo ───────────────────────────────────────────────────────────────────────

def haversine(lon1, lat1, lon2, lat2):
    R = 6371000.0
    f1, f2 = math.radians(lat1), math.radians(lat2)
    df = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a  = math.sin(df/2)**2 + math.cos(f1)*math.cos(f2)*math.sin(dl/2)**2
    return 2*R*math.asin(math.sqrt(max(0.0, min(1.0, a))))


def snap_dist(obs_lon, obs_lat, alon, alat, blon, blat):
    """Metres from GPS point to arc segment A-B."""
    mlon = haversine(obs_lon, obs_lat, obs_lon+1, obs_lat)
    mlat = haversine(obs_lon, obs_lat, obs_lon, obs_lat+1)
    ax = (alon-obs_lon)*mlon;  ay = (alat-obs_lat)*mlat
    bx = (blon-obs_lon)*mlon;  by = (blat-obs_lat)*mlat
    dx, dy = bx-ax, by-ay
    if dx == 0 and dy == 0:
        return math.hypot(ax, ay)
    t = max(0.0, min(1.0, (-ax*dx - ay*dy)/(dx*dx+dy*dy)))
    return math.hypot(ax+t*dx, ay+t*dy)


# ── loaders ───────────────────────────────────────────────────────────────────

def load_nodes(path):
    nodes = []
    with open(path) as f:
        for line in f:
            l = line.strip()
            if l:
                a, b = l.split()
                nodes.append((float(a), float(b)))
    return nodes


def load_arcs(path):
    arcs = []
    adj  = defaultdict(list)
    with open(path) as f:
        for line in f:
            l = line.strip()
            if l:
                s, d = map(int, l.split())
                aid = len(arcs)
                arcs.append((s, d))
                adj[s].append((aid, d))
    return arcs, adj


def load_track(path):
    track = []
    with open(path) as f:
        for line in f:
            l = line.strip()
            if l:
                p = l.split()
                track.append((float(p[0]), float(p[1]), float(p[2])))
    return track


def load_route(path):
    route = []
    with open(path) as f:
        for line in f:
            l = line.strip()
            if l:
                route.append(int(l))
    return route


# ── GPS subsampling ───────────────────────────────────────────────────────────

def subsample_track(track, min_dist_m=SUBSAMPLE_M):
    """Keep points that are at least min_dist_m metres apart."""
    if not track:
        return []
    out = [track[0]]
    for pt in track[1:]:
        prev = out[-1]
        if haversine(prev[0], prev[1], pt[0], pt[1]) >= min_dist_m:
            out.append(pt)
    # always include last point
    if out[-1] is not track[-1]:
        out.append(track[-1])
    return out


# ── candidate arcs ────────────────────────────────────────────────────────────

# ── spatial grid index ───────────────────────────────────────────────────────

class ArcGrid:
    """Grid-based spatial index for fast candidate arc lookup."""

    def __init__(self, nodes, arcs, cell_deg=0.005):
        self.nodes    = nodes
        self.arcs     = arcs
        self.cell_deg = cell_deg
        self.grid     = defaultdict(list)
        for aid, (s, d) in enumerate(arcs):
            lon1, lat1 = nodes[s]
            lon2, lat2 = nodes[d]
            # insert into all cells the arc's bounding box spans
            cmin_x = int(math.floor(min(lon1, lon2) / cell_deg))
            cmax_x = int(math.floor(max(lon1, lon2) / cell_deg))
            cmin_y = int(math.floor(min(lat1, lat2) / cell_deg))
            cmax_y = int(math.floor(max(lat1, lat2) / cell_deg))
            for cx in range(cmin_x, cmax_x+1):
                for cy in range(cmin_y, cmax_y+1):
                    self.grid[(cx, cy)].append(aid)

    def nearby_arcs(self, obs_lon, obs_lat, radius_deg):
        """Return set of arc IDs whose cells overlap the query radius."""
        c = self.cell_deg
        cx0 = int(math.floor((obs_lon - radius_deg) / c))
        cx1 = int(math.floor((obs_lon + radius_deg) / c))
        cy0 = int(math.floor((obs_lat - radius_deg) / c))
        cy1 = int(math.floor((obs_lat + radius_deg) / c))
        result = set()
        for cx in range(cx0, cx1+1):
            for cy in range(cy0, cy1+1):
                result.update(self.grid.get((cx, cy), []))
        return result


def find_candidates(obs_lon, obs_lat, nodes, arcs, grid=None,
                    max_cand=MAX_CAND, max_m=MAX_CAND_M):
    # approximate degree radius (1 deg ≈ 111km)
    radius_deg = max_m / 80000.0

    if grid is not None:
        candidate_aids = grid.nearby_arcs(obs_lon, obs_lat, radius_deg*2)
    else:
        candidate_aids = range(len(arcs))

    hits = []
    for aid in candidate_aids:
        s, d = arcs[aid]
        alon, alat = nodes[s]
        blon, blat = nodes[d]
        dist = snap_dist(obs_lon, obs_lat, alon, alat, blon, blat)
        if dist <= max_m:
            hits.append((dist, aid))
    hits.sort()

    if not hits:
        # widen search
        if grid is not None:
            candidate_aids = grid.nearby_arcs(obs_lon, obs_lat, radius_deg*5)
        else:
            candidate_aids = range(len(arcs))
        for aid in candidate_aids:
            s, d = arcs[aid]
            alon, alat = nodes[s]
            blon, blat = nodes[d]
            dist = snap_dist(obs_lon, obs_lat, alon, alat, blon, blat)
            hits.append((dist, aid))
        hits.sort()
        hits = hits[:3]

    return [(aid, dist) for dist, aid in hits[:max_cand]]


# ── emission ──────────────────────────────────────────────────────────────────

def emit_lp(dist_m, sigma=SIGMA_GPS):
    return -0.5*(dist_m/sigma)**2


# ── Dijkstra ──────────────────────────────────────────────────────────────────

_dijk_cache = {}

def dijkstra_node(src_node, nodes, adj, max_m=DIJK_MAX):
    key = (src_node, max_m)
    if key in _dijk_cache:
        return _dijk_cache[key]
    dist = {src_node: 0.0}
    pq   = [(0.0, src_node)]
    while pq:
        d, u = heapq.heappop(pq)
        if d > dist.get(u, INF)+1e-9:
            continue
        if d > max_m:
            break
        for _, v in adj[u]:
            w  = haversine(nodes[u][0], nodes[u][1], nodes[v][0], nodes[v][1])
            nd = d + w
            if nd < dist.get(v, INF):
                dist[v] = nd
                heapq.heappush(pq, (nd, v))
    _dijk_cache[key] = dist
    return dist


def clear_cache():
    _dijk_cache.clear()


# ── transition ────────────────────────────────────────────────────────────────

def trans_lp(road_m, gps_m, beta=BETA):
    if road_m >= INF:
        return -1e9
    return -(abs(road_m - gps_m)/beta)


# ── Viterbi ───────────────────────────────────────────────────────────────────

def viterbi_match(track, nodes, arcs, adj, subsample=True):
    """Returns list of arc IDs, one per (subsampled) GPS observation."""
    if subsample:
        track = subsample_track(track)
    n = len(track)
    if n == 0:
        return []

    # build spatial index once
    grid = ArcGrid(nodes, arcs)

    all_cands = [find_candidates(lon, lat, nodes, arcs, grid=grid)
                 for lon, lat, _ in track]

    # V[t] = {arc_id: (log_prob, prev_arc_id)}
    V = [{aid: (emit_lp(d), None) for aid, d in all_cands[0]}]

    for t in range(1, n):
        lon_p, lat_p, _ = track[t-1]
        lon_c, lat_c, _ = track[t]
        gps_m = haversine(lon_p, lat_p, lon_c, lat_c)
        radius = max(gps_m*3+300, DIJK_MAX)

        prev_alive = [(a, lp, prev) for a,(lp,prev) in V[t-1].items()
                      if lp > -1e28]

        # run Dijkstra from each prev arc's dst node (deduplicated)
        dst_nodes = {arcs[a][1] for a,_,_ in prev_alive}
        reach     = {dn: dijkstra_node(dn, nodes, adj, max_m=radius)
                     for dn in dst_nodes}

        cur = {}
        for aid, snap_m in all_cands[t]:
            em = emit_lp(snap_m)
            best_lp, best_prev = -1e30, None
            src_n = arcs[aid][0]
            for pa, prev_lp, _ in prev_alive:
                dst_n = arcs[pa][1]
                road_m = 0.0 if pa == aid else reach[dst_n].get(src_n, INF)
                lp = prev_lp + trans_lp(road_m, gps_m) + em
                if lp > best_lp:
                    best_lp, best_prev = lp, pa
            cur[aid] = (best_lp, best_prev)

        # broken chain: restart
        if not cur or max(v[0] for v in cur.values()) <= -1e28:
            cur = {aid: (emit_lp(d), None) for aid, d in all_cands[t]}

        V.append(cur)

    # traceback
    best_last = max(V[-1], key=lambda a: V[-1][a][0])
    path = [best_last]
    for t in range(n-1, 0, -1):
        prev = V[t][path[-1]][1]
        if prev is None:
            break
        path.append(prev)
    path.reverse()
    return path


# ── compress & expand route ───────────────────────────────────────────────────

def compress_route(arc_ids):
    if not arc_ids:
        return []
    out = [arc_ids[0]]
    for a in arc_ids[1:]:
        if a != out[-1]:
            out.append(a)
    return out


def arc_path(src_node, dst_node, nodes, arcs, adj, max_m=2000.0):
    """Shortest arc-path from src_node to dst_node."""
    if src_node == dst_node:
        return []
    dist      = {src_node: 0.0}
    prev_arc  = {src_node: None}
    prev_node = {src_node: None}
    pq = [(0.0, src_node)]
    found = False
    while pq:
        d, u = heapq.heappop(pq)
        if d > dist.get(u, INF)+1e-9:
            continue
        if u == dst_node:
            found = True; break
        if d > max_m:
            break
        for aid, v in adj[u]:
            w  = haversine(nodes[u][0], nodes[u][1], nodes[v][0], nodes[v][1])
            nd = d + w
            if nd < dist.get(v, INF):
                dist[v]     = nd
                prev_arc[v] = aid
                prev_node[v]= u
                heapq.heappush(pq, (nd, v))
    if not found:
        return []
    path, cur = [], dst_node
    while prev_node.get(cur) is not None:
        path.append(prev_arc[cur])
        cur = prev_node[cur]
    return list(reversed(path))


def expand_route(arc_ids, nodes, arcs, adj):
    """Fill gaps between consecutive non-connected arcs."""
    if len(arc_ids) < 2:
        return list(arc_ids)
    full = [arc_ids[0]]
    for i in range(1, len(arc_ids)):
        pa, ca = arc_ids[i-1], arc_ids[i]
        prev_dst = arcs[pa][1]
        curr_src = arcs[ca][0]
        if prev_dst != curr_src:
            mid = arc_path(prev_dst, curr_src, nodes, arcs, adj)
            full.extend(mid)
        full.append(ca)
    return compress_route(full)


# ── evaluation ────────────────────────────────────────────────────────────────

def route_overlap_score(predicted, ground_truth):
    if not predicted and not ground_truth:
        return 1.0, 1.0, 1.0
    if not predicted or not ground_truth:
        return 0.0, 0.0, 0.0
    ps = set(predicted); gs = set(ground_truth)
    tp   = len(ps & gs)
    prec = tp/len(ps) if ps else 0.0
    rec  = tp/len(gs) if gs else 0.0
    f1   = 2*prec*rec/(prec+rec) if (prec+rec) > 0 else 0.0
    return prec, rec, f1


# ── process one scenario ──────────────────────────────────────────────────────

def process_scenario(folder, scenario_id, has_ground_truth=True):
    base       = os.path.join(folder, scenario_id)
    nodes      = load_nodes(base + '.nodes')
    arcs, adj  = load_arcs(base + '.arcs')
    track      = load_track(base + '.track')

    clear_cache()
    matched    = viterbi_match(track, nodes, arcs, adj, subsample=True)
    compressed = compress_route(matched)
    predicted  = expand_route(compressed, nodes, arcs, adj)

    metrics = None
    if has_ground_truth and os.path.exists(base + '.route'):
        gt = load_route(base + '.route')
        p, r, f1 = route_overlap_score(predicted, gt)
        metrics = {'precision': p, 'recall': r, 'f1': f1,
                   'pred_len': len(predicted), 'gt_len': len(gt)}
    return predicted, metrics


# ── standalone entry ──────────────────────────────────────────────────────────

if __name__ == '__main__':
    import sys

    base_dir  = os.path.dirname(os.path.abspath(__file__))
    workspace = os.path.dirname(base_dir)
    eval_dir  = os.path.join(workspace, 'evaluator_test', 'evaluator_test')
    stud_dir  = os.path.join(workspace, 'Student folder', 'Student folder', 'student_test')
    out_dir   = os.path.join(base_dir, 'predictions')
    os.makedirs(out_dir, exist_ok=True)

    if os.path.isdir(eval_dir):
        print("\n[Evaluator test]")
        f1s = []
        for sid in sorted(d for d in os.listdir(eval_dir)
                          if os.path.isdir(os.path.join(eval_dir, d))):
            pred, m = process_scenario(os.path.join(eval_dir, sid), sid)
            if m:
                f1s.append(m['f1'])
                print(f"  {sid}: P={m['precision']:.3f} R={m['recall']:.3f} F1={m['f1']:.3f}")
            with open(os.path.join(out_dir, sid+'.route'), 'w') as f:
                f.write('\n'.join(map(str, pred))+'\n')
        if f1s:
            import numpy as np
            print(f"  Mean F1 = {np.mean(f1s):.4f}")

    if os.path.isdir(stud_dir):
        print("\n[Student test]")
        for sid in sorted(d for d in os.listdir(stud_dir)
                          if os.path.isdir(os.path.join(stud_dir, d))):
            pred, _ = process_scenario(os.path.join(stud_dir, sid), sid,
                                       has_ground_truth=False)
            out = os.path.join(out_dir, f'student_{sid}.route')
            with open(out, 'w') as f:
                f.write('\n'.join(map(str, pred))+'\n')
            print(f"  {sid}: {len(pred)} arcs")
