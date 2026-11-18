"""
PathSnap Demo - single scenario, end-to-end with plot.
Usage:  python demo.py [scenario_id]
e.g.:   python demo.py 00000015
"""
import os, sys, time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from map_matching import (load_nodes, load_arcs, load_track, load_route,
                          viterbi_match, compress_route, expand_route,
                          route_overlap_score, haversine)

# ── pick scenario ─────────────────────────────────────────────────────────────
base_dir  = os.path.dirname(os.path.abspath(__file__))
workspace = os.path.dirname(base_dir)
SID       = sys.argv[1] if len(sys.argv) > 1 else '00000015'

# try evaluator_test first, then student_test
folder = os.path.join(workspace, 'evaluator_test', 'evaluator_test', SID)
if not os.path.isdir(folder):
    folder = os.path.join(workspace, 'Student folder', 'Student folder', 'student_test', SID)

print(f"Scenario : {SID}")
print(f"Folder   : {folder}")

base  = os.path.join(folder, SID)
nodes = load_nodes(base + '.nodes')
arcs, adj = load_arcs(base + '.arcs')
track = load_track(base + '.track')
has_gt = os.path.exists(base + '.route')
gt     = load_route(base + '.route') if has_gt else []

print(f"Nodes={len(nodes)}  Arcs={len(arcs)}  GPS pts={len(track)}  GT arcs={len(gt)}")

# ── run matching ──────────────────────────────────────────────────────────────
print("Running HMM Viterbi...", flush=True)
t0 = time.time()
matched    = viterbi_match(track, nodes, arcs, adj)
compressed = compress_route(matched)
predicted  = expand_route(compressed, nodes, arcs, adj)
elapsed    = time.time() - t0
print(f"Done in {elapsed:.1f}s  |  predicted {len(predicted)} arcs")

p = r = f1 = None
if has_gt:
    p, r, f1 = route_overlap_score(predicted, gt)
    print(f"Precision={p:.4f}  Recall={r:.4f}  F1={f1:.4f}")

# ── helpers ───────────────────────────────────────────────────────────────────
def arc_coords(arc_id):
    s, d = arcs[arc_id]
    return nodes[s], nodes[d]

def arc_lons_lats(arc_ids):
    lons, lats = [], []
    for a in arc_ids:
        (lon1,lat1),(lon2,lat2) = arc_coords(a)
        lons += [lon1, lon2, None]
        lats += [lat1, lat2, None]
    return lons, lats

# ── plot ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(18, 8))
title_base = f'PathSnap – Scenario {SID}'
if has_gt and f1 is not None:
    title_base += f'   |   F1={f1:.3f}'
fig.suptitle(title_base, fontsize=13, fontweight='bold')

for ax_i, ax in enumerate(axes):
    # road network (light)
    for s, d in arcs:
        ax.plot([nodes[s][0], nodes[d][0]],
                [nodes[s][1], nodes[d][1]],
                color='#dddddd', lw=0.4, zorder=1)

    # GPS track
    gx = [p[0] for p in track]
    gy = [p[1] for p in track]
    ax.scatter(gx, gy, c='dodgerblue', s=3, alpha=0.45, zorder=3,
               label='GPS track')

    if ax_i == 0:
        ax.set_title('Input: Road Network + GPS Track', fontsize=10)
    else:
        # predicted route
        if predicted:
            px, py = arc_lons_lats(predicted)
            ax.plot(px, py, color='red', lw=2.2, zorder=5, label='Predicted')
            # start / end markers
            (s0lon, s0lat), _ = arc_coords(predicted[0])
            _, (elon, elat)   = arc_coords(predicted[-1])
            ax.scatter([s0lon], [s0lat], s=120, color='lime',  zorder=6, label='Start')
            ax.scatter([elon],  [elat],  s=120, color='purple',zorder=6, label='End')

        # ground truth
        if has_gt and gt:
            gx2, gy2 = arc_lons_lats(gt)
            ax.plot(gx2, gy2, color='green', lw=1.8, ls='--', alpha=0.75,
                    zorder=4, label='Ground truth')

        label = 'Output: Predicted Route'
        if has_gt and f1 is not None:
            label += f'\nP={p:.3f}  R={r:.3f}  F1={f1:.3f}'
        ax.set_title(label, fontsize=10)
        ax.legend(fontsize=8, loc='best')

    ax.set_xlabel('Longitude'); ax.set_ylabel('Latitude')
    ax.set_aspect('equal'); ax.grid(True, alpha=0.25)

plt.tight_layout()
out_img = os.path.join(base_dir, f'demo_{SID}.png')
plt.savefig(out_img, dpi=150, bbox_inches='tight')
print(f"Plot saved → {out_img}")
plt.close()

# ── summary ───────────────────────────────────────────────────────────────────
print("\n" + "="*52)
print("SUMMARY")
print("="*52)
print(f"  Scenario          : {SID}")
print(f"  Road nodes        : {len(nodes)}")
print(f"  Road arcs         : {len(arcs)}")
print(f"  GPS observations  : {len(track)}")
print(f"  Predicted arcs    : {len(predicted)}")
if has_gt:
    print(f"  Ground-truth arcs : {len(gt)}")
    print(f"  Precision         : {p:.4f}")
    print(f"  Recall            : {r:.4f}")
    print(f"  F1 Score          : {f1:.4f}")
print(f"  Time              : {elapsed:.2f}s")
print(f"  Plot              : demo_{SID}.png")
print("="*52)
