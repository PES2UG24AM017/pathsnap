"""
eda.py  –  Exploratory Data Analysis for PathSnap dataset
==========================================================
Produces:
  - Dataset inventory (trip counts, file counts, observation counts)
  - GPS stats (sampling interval, speed, coordinate range)
  - Road network stats (nodes, arcs, connectivity)
  - GPS trajectory plots (matplotlib)
  - Folium interactive maps (eda_folium_trip.html)
"""

import os, sys, math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import folium
from folium.plugins import MeasureControl

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from map_matching import load_nodes, load_arcs, load_track, load_route, haversine

base_dir  = os.path.dirname(os.path.abspath(__file__))
workspace = os.path.dirname(base_dir)

PARTITIONS = {
    'train'       : os.path.join(workspace, 'Student folder', 'Student folder', 'train'),
    'student_test': os.path.join(workspace, 'Student folder', 'Student folder', 'student_test'),
    'eval_test'   : os.path.join(workspace, 'evaluator_test', 'evaluator_test'),
}

# ── inventory ─────────────────────────────────────────────────────────────────
print("="*65)
print("PathSnap – Dataset Inventory")
print("="*65)
print(f"{'Partition':<15} {'Trips':>6} {'GPS pts (mean)':>15} {'Nodes (mean)':>13} {'Arcs (mean)':>12}")
print("-"*65)

all_stats = {}
for part, folder in PARTITIONS.items():
    if not os.path.isdir(folder):
        continue
    sids = sorted([d for d in os.listdir(folder) if os.path.isdir(os.path.join(folder, d))])
    gps_counts, node_counts, arc_counts = [], [], []
    for sid in sids:
        base = os.path.join(folder, sid, sid)
        try:
            t = load_track(base + '.track')
            n = load_nodes(base + '.nodes')
            with open(base + '.arcs') as f:
                a = sum(1 for l in f if l.strip())
            gps_counts.append(len(t))
            node_counts.append(len(n))
            arc_counts.append(a)
        except:
            pass
    all_stats[part] = {'sids': sids, 'gps': gps_counts,
                       'nodes': node_counts, 'arcs': arc_counts}
    print(f"{part:<15} {len(sids):>6} "
          f"{np.mean(gps_counts):>15.0f} "
          f"{np.mean(node_counts):>13.0f} "
          f"{np.mean(arc_counts):>12.0f}")

print("="*65)

# ── GPS statistics ─────────────────────────────────────────────────────────────
print("\n--- GPS Sampling Interval Analysis (train partition) ---")
train_folder = PARTITIONS['train']
sids = all_stats['train']['sids'][:10]   # sample 10 trips
intervals_all = []
speeds_all    = []
for sid in sids:
    base  = os.path.join(train_folder, sid, sid)
    track = load_track(base + '.track')
    for i in range(1, len(track)):
        dt   = track[i][2] - track[i-1][2]
        dist = haversine(track[i-1][0], track[i-1][1], track[i][0], track[i][1])
        intervals_all.append(dt)
        if dt > 0:
            speeds_all.append(dist/dt * 3.6)   # km/h

intervals_all = np.array(intervals_all)
speeds_all    = np.array(speeds_all)
print(f"  Sampling interval: mean={intervals_all.mean():.2f}s  "
      f"median={np.median(intervals_all):.2f}s  "
      f"max={intervals_all.max():.0f}s")
print(f"  GPS speed: mean={speeds_all.mean():.1f} km/h  "
      f"max={speeds_all.max():.1f} km/h  "
      f"outliers (>120km/h): {(speeds_all>120).sum()}")

# ── Missing-value analysis ─────────────────────────────────────────────────────
print("\n--- Missing-value / Anomaly Analysis ---")
large_gaps = sum(1 for i in intervals_all if i > 10)
print(f"  GPS gaps > 10s: {large_gaps} ({100*large_gaps/len(intervals_all):.1f}% of steps)")
print(f"  GPS gaps > 30s: {sum(1 for i in intervals_all if i>30)}")

# ── plots ─────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Sampling interval histogram
axes[0].hist(intervals_all[intervals_all <= 10], bins=30, color='steelblue', edgecolor='white')
axes[0].set_xlabel('Sampling interval (s)'); axes[0].set_ylabel('Count')
axes[0].set_title('GPS Sampling Intervals (≤10s)')
axes[0].grid(alpha=0.3)

# Speed histogram
axes[1].hist(speeds_all[speeds_all < 120], bins=40, color='darkorange', edgecolor='white')
axes[1].set_xlabel('Speed (km/h)'); axes[1].set_ylabel('Count')
axes[1].set_title('GPS Speed Distribution')
axes[1].grid(alpha=0.3)

# GPS points per trip
all_gps = all_stats['train']['gps'] + all_stats.get('student_test',{}).get('gps',[])
axes[2].hist(all_gps, bins=20, color='green', edgecolor='white')
axes[2].set_xlabel('GPS points per trip'); axes[2].set_ylabel('Count')
axes[2].set_title('GPS Points per Trip')
axes[2].grid(alpha=0.3)

plt.suptitle('PathSnap – EDA: GPS Statistics', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(base_dir, 'eda_gps_stats.png'), dpi=150)
plt.close()
print("\nSaved: eda_gps_stats.png")

# ── road-network stats ────────────────────────────────────────────────────────
# pick one representative trip
sid_eg = sids[0]
base_eg = os.path.join(train_folder, sid_eg, sid_eg)
nodes_eg = load_nodes(base_eg + '.nodes')
arcs_eg, adj_eg = load_arcs(base_eg + '.arcs')
out_degrees = [len(adj_eg[n]) for n in range(len(nodes_eg))]
print(f"\n--- Road Network ({sid_eg}) ---")
print(f"  Nodes: {len(nodes_eg)}   Arcs: {len(arcs_eg)}")
print(f"  Out-degree: mean={np.mean(out_degrees):.2f}  max={max(out_degrees)}  "
      f"isolated(0)={sum(1 for d in out_degrees if d==0)}")

# ── GPS trajectory plot ────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
example_sids = sids[:3]
for ax, sid in zip(axes, example_sids):
    base = os.path.join(train_folder, sid, sid)
    nodes  = load_nodes(base + '.nodes')
    arcs_l, _ = load_arcs(base + '.arcs')
    track  = load_track(base + '.track')
    gt     = load_route(base + '.route') if os.path.exists(base+'.route') else []
    # road network
    for s, d in arcs_l:
        ax.plot([nodes[s][0], nodes[d][0]], [nodes[s][1], nodes[d][1]],
                color='#dddddd', lw=0.5)
    # GPS track
    ax.scatter([p[0] for p in track], [p[1] for p in track],
               c='dodgerblue', s=2, alpha=0.4, label='GPS')
    # GT route
    if gt:
        gxs, gys = [], []
        for a in gt:
            s, d = arcs_l[a]
            gxs += [nodes[s][0], nodes[d][0], None]
            gys += [nodes[s][1], nodes[d][1], None]
        ax.plot(gxs, gys, 'g-', lw=2, alpha=0.8, label='Ground truth')
    ax.set_title(f'Trip {sid}', fontsize=9)
    ax.set_aspect('equal'); ax.grid(alpha=0.2)
    ax.legend(fontsize=7)
plt.suptitle('PathSnap – GPS Trajectories & Ground Truth Routes', fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(base_dir, 'eda_trajectories.png'), dpi=150)
plt.close()
print("Saved: eda_trajectories.png")

# ── Folium interactive map ────────────────────────────────────────────────────
print("\nGenerating Folium interactive map...")
sid_f = sids[0]
base_f = os.path.join(train_folder, sid_f, sid_f)
nodes_f = load_nodes(base_f + '.nodes')
arcs_f, _ = load_arcs(base_f + '.arcs')
track_f   = load_track(base_f + '.track')
gt_f      = load_route(base_f + '.route') if os.path.exists(base_f+'.route') else []

# centre map
clat = np.mean([p[1] for p in track_f])
clon = np.mean([p[0] for p in track_f])
m = folium.Map(location=[clat, clon], zoom_start=15, tiles='OpenStreetMap')

# road network layer
road_fg = folium.FeatureGroup(name='Road Network', show=True)
for s, d in arcs_f:
    folium.PolyLine(
        [[nodes_f[s][1], nodes_f[s][0]], [nodes_f[d][1], nodes_f[d][0]]],
        color='#888888', weight=1.5, opacity=0.5
    ).add_to(road_fg)
road_fg.add_to(m)

# GPS track layer
gps_fg = folium.FeatureGroup(name='GPS Track (raw)', show=True)
gps_coords = [[p[1], p[0]] for p in track_f[::5]]   # downsample for display
folium.PolyLine(gps_coords, color='blue', weight=2, opacity=0.6).add_to(gps_fg)
# mark each GPS point (further downsampled)
for p in track_f[::20]:
    folium.CircleMarker([p[1], p[0]], radius=3,
                        color='blue', fill=True, fill_opacity=0.5).add_to(gps_fg)
gps_fg.add_to(m)

# ground truth route layer
if gt_f:
    gt_fg = folium.FeatureGroup(name='Ground Truth Route', show=True)
    gt_coords = []
    for a in gt_f:
        s, d = arcs_f[a]
        gt_coords += [[nodes_f[s][1], nodes_f[s][0]],
                      [nodes_f[d][1], nodes_f[d][0]]]
    folium.PolyLine(gt_coords, color='green', weight=4, opacity=0.8).add_to(gt_fg)
    gt_fg.add_to(m)

    # start/end markers
    s0, _ = arcs_f[gt_f[0]]
    _,  de = arcs_f[gt_f[-1]]
    folium.Marker([nodes_f[s0][1], nodes_f[s0][0]],
                  popup='Start', icon=folium.Icon(color='green', icon='play')).add_to(m)
    folium.Marker([nodes_f[de][1], nodes_f[de][0]],
                  popup='End', icon=folium.Icon(color='red', icon='stop')).add_to(m)

folium.LayerControl().add_to(m)
MeasureControl().add_to(m)
folium.map.Tooltip(f'Trip: {sid_f}').add_to(m)

out_html = os.path.join(base_dir, 'eda_folium_trip.html')
m.save(out_html)
print(f"Saved: eda_folium_trip.html  (open in browser)")
print("\nEDA complete.")
