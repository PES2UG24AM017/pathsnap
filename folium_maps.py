"""
folium_maps.py  –  Generate interactive Folium maps for PathSnap
================================================================
Creates one HTML map per scenario showing:
  - Road network (grey)
  - Raw GPS track (blue dots)
  - Predicted route (red line)
  - Ground truth route (green dashed, if available)
  - Start / End markers

Usage:
  python folium_maps.py [scenario_id]   # single scenario
  python folium_maps.py --all           # all evaluator scenarios
"""

import os, sys
import numpy as np
import folium
from folium.plugins import MeasureControl, MiniMap

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from map_matching import (load_nodes, load_arcs, load_track, load_route,
                          viterbi_match, compress_route, expand_route,
                          route_overlap_score)

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
WORKSPACE = os.path.dirname(BASE_DIR)
EVAL_DIR  = os.path.join(WORKSPACE, 'evaluator_test', 'evaluator_test')
OUT_DIR   = os.path.join(BASE_DIR,  'maps')
os.makedirs(OUT_DIR, exist_ok=True)

PRED_DIR = os.path.join(BASE_DIR, 'predictions')


def build_map(scenario_id, folder=None):
    if folder is None:
        folder = os.path.join(EVAL_DIR, scenario_id)

    base      = os.path.join(folder, scenario_id)
    nodes     = load_nodes(base + '.nodes')
    arcs, adj = load_arcs(base + '.arcs')
    track     = load_track(base + '.track')
    has_gt    = os.path.exists(base + '.route')
    gt        = load_route(base + '.route') if has_gt else []

    # load or compute predicted route
    pred_path = os.path.join(PRED_DIR, scenario_id + '.route')
    if os.path.exists(pred_path):
        predicted = [int(l.strip()) for l in open(pred_path)
                     if l.strip()]
    else:
        matched   = viterbi_match(track, nodes, arcs, adj, subsample=True)
        compressed = compress_route(matched)
        predicted  = expand_route(compressed, nodes, arcs, adj)

    # metrics
    if has_gt and gt:
        p, r, f1 = route_overlap_score(predicted, gt)
    else:
        p = r = f1 = None

    # map centre
    clat = np.mean([pt[1] for pt in track])
    clon = np.mean([pt[0] for pt in track])

    m = folium.Map(location=[clat, clon], zoom_start=15,
                   tiles='CartoDB positron')

    # ── road network layer ────────────────────────────────────────────────────
    road_fg = folium.FeatureGroup(name='Road Network', show=True)
    for s, d in arcs:
        folium.PolyLine(
            [[nodes[s][1], nodes[s][0]], [nodes[d][1], nodes[d][0]]],
            color='#aaaaaa', weight=1.2, opacity=0.5,
        ).add_to(road_fg)
    road_fg.add_to(m)

    # ── GPS track layer ───────────────────────────────────────────────────────
    gps_fg = folium.FeatureGroup(name='GPS Track (raw)', show=True)
    gps_latlons = [[pt[1], pt[0]] for pt in track]
    folium.PolyLine(gps_latlons, color='#2196F3', weight=2,
                    opacity=0.5).add_to(gps_fg)
    # sampled GPS circles
    step = max(1, len(track) // 150)
    for pt in track[::step]:
        folium.CircleMarker(
            location=[pt[1], pt[0]], radius=3,
            color='#2196F3', fill=True, fill_color='#2196F3',
            fill_opacity=0.6,
            tooltip=f"lon={pt[0]:.5f} lat={pt[1]:.5f} t={pt[2]:.0f}s"
        ).add_to(gps_fg)
    gps_fg.add_to(m)

    # ── predicted route layer ─────────────────────────────────────────────────
    if predicted:
        pred_fg = folium.FeatureGroup(name='Predicted Route', show=True)
        pred_coords = []
        for a in predicted:
            s, d = arcs[a]
            pred_coords += [[nodes[s][1], nodes[s][0]],
                            [nodes[d][1], nodes[d][0]]]
        folium.PolyLine(pred_coords, color='#F44336', weight=4,
                        opacity=0.85).add_to(pred_fg)

        # start / end markers
        s0, _  = arcs[predicted[0]]
        _, de  = arcs[predicted[-1]]
        folium.Marker(
            [nodes[s0][1], nodes[s0][0]], popup='<b>Predicted Start</b>',
            icon=folium.Icon(color='green', icon='play', prefix='fa')
        ).add_to(pred_fg)
        folium.Marker(
            [nodes[de][1], nodes[de][0]], popup='<b>Predicted End</b>',
            icon=folium.Icon(color='red', icon='stop', prefix='fa')
        ).add_to(pred_fg)
        pred_fg.add_to(m)

    # ── ground truth layer ────────────────────────────────────────────────────
    if has_gt and gt:
        gt_fg = folium.FeatureGroup(name='Ground Truth Route', show=True)
        gt_coords = []
        for a in gt:
            s, d = arcs[a]
            gt_coords += [[nodes[s][1], nodes[s][0]],
                          [nodes[d][1], nodes[d][0]]]
        folium.PolyLine(gt_coords, color='#4CAF50', weight=3.5,
                        dash_array='8 4', opacity=0.9).add_to(gt_fg)
        gt_fg.add_to(m)

    # ── controls and title ────────────────────────────────────────────────────
    folium.LayerControl(collapsed=False).add_to(m)
    MeasureControl().add_to(m)
    MiniMap(toggle_display=True).add_to(m)

    metric_str = ""
    if f1 is not None:
        metric_str = f" | P={p:.3f} R={r:.3f} F1={f1:.3f}"

    title_html = f"""
    <div style="position:fixed;top:10px;left:60px;z-index:1000;
                background:white;padding:8px 12px;border-radius:6px;
                box-shadow:0 2px 6px rgba(0,0,0,.3);font-family:Arial;font-size:13px;">
        <b>PathSnap</b> – Trip {scenario_id}{metric_str}<br>
        <span style="color:#2196F3">&#9679;</span> GPS track &nbsp;
        <span style="color:#F44336">&#9644;</span> Predicted &nbsp;
        <span style="color:#4CAF50">&#9644;</span> Ground truth
    </div>"""
    m.get_root().html.add_child(folium.Element(title_html))

    out_path = os.path.join(OUT_DIR, f'map_{scenario_id}.html')
    m.save(out_path)
    print(f"  {scenario_id}: saved → {out_path}"
          + (f"  F1={f1:.4f}" if f1 is not None else ""))
    return out_path


# ── main ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    args = sys.argv[1:]

    if '--all' in args or not args:
        # all evaluator scenarios
        scenarios = sorted([d for d in os.listdir(EVAL_DIR)
                            if os.path.isdir(os.path.join(EVAL_DIR, d))])
        print(f"Generating {len(scenarios)} Folium maps...")
        for sid in scenarios:
            build_map(sid)
        print(f"\nAll maps saved to: {OUT_DIR}")
    else:
        sid = args[0]
        path = build_map(sid)
        print(f"Open in browser: {path}")
