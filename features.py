"""
features.py  –  Feature engineering for PathSnap candidate scoring
===================================================================
For every GPS observation we generate candidate arcs and extract
features that describe how well each arc matches the observation.

Feature vector (per candidate):
  0  snap_dist_m        – perpendicular distance GPS → arc (metres)
  1  snap_dist_norm     – snap_dist / sigma_gps
  2  bearing_diff       – |GPS heading – arc heading| in [0,180]
  3  arc_len_m          – arc length (metres)
  4  gps_speed_mps      – estimated GPS speed (m/s) at this point
  5  time_delta_s       – seconds since previous GPS point
  6  gps_to_prev_m      – GPS displacement since previous point
  7  road_dist_to_prev  – Dijkstra road dist from prev best arc's dst to this arc's src
  8  dist_ratio         – road_dist_to_prev / (gps_to_prev_m + 1e-6)
  9  cand_rank          – rank of this candidate (0 = closest)
 10  cand_total         – total candidates for this observation
 11  norm_snap          – snap rank / total candidates

Label: 1 if this arc_id matches any arc_id in the ground truth route for this trip, else 0
       For the Top-1 task: label is 1 for the single best matching arc.
"""

import os, math
import numpy as np
from map_matching import (load_nodes, load_arcs, load_track, load_route,
                          subsample_track, find_candidates, snap_dist,
                          haversine, ArcGrid, dijkstra_node, INF)

SIGMA = 20.0

# ── bearing helpers ───────────────────────────────────────────────────────────

def bearing(lon1, lat1, lon2, lat2):
    """Compass bearing in degrees [0,360)."""
    dl = math.radians(lon2 - lon1)
    f1, f2 = math.radians(lat1), math.radians(lat2)
    x = math.sin(dl)*math.cos(f2)
    y = math.cos(f1)*math.sin(f2) - math.sin(f1)*math.cos(f2)*math.cos(dl)
    b = math.degrees(math.atan2(x, y))
    return (b + 360) % 360


def bearing_diff(b1, b2):
    """Absolute angular difference in [0,180]."""
    d = abs(b1 - b2) % 360
    return d if d <= 180 else 360 - d


# ── build feature matrix for one trip ────────────────────────────────────────

def extract_features(folder, scenario_id, subsample=True):
    """
    Returns:
      X  – np.array shape (n_candidates_total, 12)
      y  – np.array shape (n_candidates_total,)   label: arc in GT route?
      groups – np.array shape (n_candidates_total,)  observation index
      obs_arc_ids – list of list[arc_id] per observation
    """
    base       = os.path.join(folder, scenario_id)
    nodes      = load_nodes(base + '.nodes')
    arcs, adj  = load_arcs(base + '.arcs')
    track      = load_track(base + '.track')
    has_route  = os.path.exists(base + '.route')
    gt_set     = set(load_route(base + '.route')) if has_route else set()

    if subsample:
        track = subsample_track(track)

    grid = ArcGrid(nodes, arcs)

    # arc lengths (cached)
    arc_len = []
    for s, d in arcs:
        arc_len.append(haversine(nodes[s][0], nodes[s][1], nodes[d][0], nodes[d][1]))

    rows, labels, groups = [], [], []
    obs_arc_ids = []

    prev_lon, prev_lat, prev_ts = None, None, None
    # Dijkstra cache: dst node of previous best arc → reachable distances
    prev_reach = {}
    prev_best_arc = None

    for obs_i, (lon, lat, ts) in enumerate(track):
        cands = find_candidates(lon, lat, nodes, arcs, grid=grid)
        obs_arc_ids.append([a for a, _ in cands])

        # GPS kinematics
        if prev_lon is None:
            gps_disp  = 0.0
            dt        = 1.0
            gps_speed = 0.0
            gps_bear  = 0.0
        else:
            gps_disp  = haversine(prev_lon, prev_lat, lon, lat)
            dt        = max(ts - prev_ts, 1.0)
            gps_speed = gps_disp / dt
            gps_bear  = bearing(prev_lon, prev_lat, lon, lat) if gps_disp > 1 else 0.0

        for rank, (aid, snap_m) in enumerate(cands):
            s, d = arcs[aid]
            arc_b    = bearing(nodes[s][0], nodes[s][1], nodes[d][0], nodes[d][1])
            b_diff   = bearing_diff(gps_bear, arc_b)
            a_len    = arc_len[aid]
            src_node = s

            # road distance from prev best arc dst → this arc src
            if prev_best_arc is not None and prev_reach:
                road_d = prev_reach.get(src_node, INF)
                if road_d == INF:
                    road_d = gps_disp * 2
            else:
                road_d = 0.0
            dist_ratio = road_d / (gps_disp + 1e-6)

            feat = [
                snap_m,
                snap_m / SIGMA,
                b_diff,
                a_len,
                gps_speed,
                dt,
                gps_disp,
                road_d,
                min(dist_ratio, 10.0),
                rank,
                len(cands),
                rank / max(len(cands)-1, 1),
            ]
            rows.append(feat)
            labels.append(1 if aid in gt_set else 0)
            groups.append(obs_i)

        # update prev
        prev_lon, prev_lat, prev_ts = lon, lat, ts
        # run Dijkstra from closest candidate's dst for next step
        if cands:
            best_aid = cands[0][0]
            dst_node = arcs[best_aid][1]
            prev_reach     = dijkstra_node(dst_node, nodes, adj)
            prev_best_arc  = best_aid

    X = np.array(rows,   dtype=np.float32)
    y = np.array(labels, dtype=np.int32)
    g = np.array(groups, dtype=np.int32)
    return X, y, g, obs_arc_ids


# ── build full dataset from a folder of scenarios ────────────────────────────

def build_dataset(data_dir, max_trips=None, verbose=True):
    """
    Load all trips in data_dir, extract features, return (X, y, groups, trip_ids).
    groups here = trip_index * 10000 + obs_index  (unique per observation).
    """
    scenarios = sorted([d for d in os.listdir(data_dir)
                        if os.path.isdir(os.path.join(data_dir, d))])
    if max_trips:
        scenarios = scenarios[:max_trips]

    all_X, all_y, all_g = [], [], []
    trip_id_col = []
    obs_offset  = 0

    for tid, sid in enumerate(scenarios):
        folder = os.path.join(data_dir, sid)
        if not os.path.exists(os.path.join(folder, sid + '.route')):
            continue   # skip if no ground truth
        try:
            X, y, g, _ = extract_features(folder, sid)
            # make groups unique across trips
            all_X.append(X)
            all_y.append(y)
            all_g.append(g + obs_offset)
            trip_id_col.extend([tid] * len(y))
            obs_offset += g.max() + 1 if len(g) > 0 else 0
            if verbose:
                pos = y.sum()
                print(f"  {sid}: {len(X)} candidates, "
                      f"{pos} positive ({100*pos/max(len(y),1):.1f}%)")
        except Exception as e:
            if verbose:
                print(f"  {sid}: ERROR – {e}")

    X_all = np.vstack(all_X)
    y_all = np.concatenate(all_y)
    g_all = np.concatenate(all_g)
    return X_all, y_all, g_all, np.array(trip_id_col)


FEATURE_NAMES = [
    'snap_dist_m', 'snap_dist_norm', 'bearing_diff', 'arc_len_m',
    'gps_speed_mps', 'time_delta_s', 'gps_to_prev_m', 'road_dist_to_prev',
    'dist_ratio', 'cand_rank', 'cand_total', 'norm_snap',
]
