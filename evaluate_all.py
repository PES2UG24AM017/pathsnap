"""
evaluate_all.py - Run map-matching on ALL evaluator_test scenarios.
Prints per-scenario metrics + aggregate, saves a bar-chart PNG.
"""
import os, sys, time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from map_matching import process_scenario

base_dir  = os.path.dirname(os.path.abspath(__file__))
workspace = os.path.dirname(base_dir)
eval_dir  = os.path.join(workspace, 'evaluator_test', 'evaluator_test')
out_dir   = os.path.join(base_dir, 'predictions')
os.makedirs(out_dir, exist_ok=True)

scenarios = sorted([d for d in os.listdir(eval_dir)
                    if os.path.isdir(os.path.join(eval_dir, d))])

print("="*62)
print("PathSnap - Evaluator Test Set")
print("="*62)
print(f"{'Scenario':<12} {'P':>6} {'R':>6} {'F1':>6} {'Pred':>5} {'GT':>5} {'t(s)':>5}")
print("-"*62)

rows = []
t_all = time.time()
for sid in scenarios:
    folder = os.path.join(eval_dir, sid)
    t0 = time.time()
    pred, m = process_scenario(folder, sid, has_ground_truth=True)
    dt = time.time() - t0
    # save
    with open(os.path.join(out_dir, sid+'.route'), 'w') as f:
        f.write('\n'.join(map(str, pred))+'\n')
    if m:
        rows.append({'id': sid, 'dt': dt, **m})
        print(f"{sid:<12} {m['precision']:>6.3f} {m['recall']:>6.3f} "
              f"{m['f1']:>6.3f} {m['pred_len']:>5} {m['gt_len']:>5} {dt:>5.1f}")

total_t = time.time() - t_all

precs = [r['precision'] for r in rows]
recs  = [r['recall']    for r in rows]
f1s   = [r['f1']        for r in rows]

print("="*62)
print(f"{'Mean':<12} {np.mean(precs):>6.3f} {np.mean(recs):>6.3f} {np.mean(f1s):>6.3f}")
print(f"{'Std':<12} {np.std(precs):>6.3f} {np.std(recs):>6.3f} {np.std(f1s):>6.3f}")
print(f"\nTotal time: {total_t:.1f}s")
print(f"Predictions saved to: {out_dir}")

# ── bar chart ─────────────────────────────────────────────────────────────────
ids = [r['id'] for r in rows]
x   = np.arange(len(ids))
w   = 0.25

fig, ax = plt.subplots(figsize=(12, 5))
ax.bar(x-w, precs, w, label='Precision', color='steelblue')
ax.bar(x,   recs,  w, label='Recall',    color='darkorange')
ax.bar(x+w, f1s,   w, label='F1',        color='green')
ax.axhline(np.mean(f1s), color='green', ls='--', lw=1.2, alpha=0.6)
ax.set_xticks(x); ax.set_xticklabels(ids, rotation=40, ha='right', fontsize=9)
ax.set_ylim(0, 1.05); ax.set_ylabel('Score')
ax.set_title(f'PathSnap Evaluation Results  |  Mean F1 = {np.mean(f1s):.4f}')
ax.legend(); ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
out_img = os.path.join(base_dir, 'evaluation_results.png')
plt.savefig(out_img, dpi=150)
print(f"Plot saved → {out_img}")
