"""
compare_clf_hmm.py
Classifier-only  vs  HMM-only  vs  Classifier+HMM on the evaluator trips.
Also reports Route Continuity and Candidate Miss Rate.

Outputs: clf_vs_hmm_comparison.png, clf_vs_hmm_results.csv
(set PATHSNAP_EVAL_DIR to point at the evaluator folder if it is elsewhere)
"""
import os, sys, csv
import numpy as np
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pipeline import run_trip

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
EVAL_DIR  = os.environ.get('PATHSNAP_EVAL_DIR',
            os.path.join(os.path.dirname(BASE_DIR), 'evaluator_test', 'evaluator_test'))
clf = joblib.load(os.path.join(BASE_DIR, 'models', 'best_model.joblib'))
print(f"Loaded model: {type(clf).__name__}   eval dir: {EVAL_DIR}")

M = ['classifier_only', 'hmm_only', 'classifier_hmm']
NAMES = {'classifier_only': 'Classifier only', 'hmm_only': 'HMM only',
         'classifier_hmm': 'Classifier + HMM'}
scenarios = sorted(d for d in os.listdir(EVAL_DIR)
                   if os.path.isdir(os.path.join(EVAL_DIR, d)))

rows = []
print("\n" + "=" * 100)
print(f"{'Scenario':<10} | {'F1: Clf':>7} {'HMM':>7} {'C+H':>7} | "
      f"{'Top1: Clf':>9} {'HMM':>6} {'C+H':>6} | {'Cont%: Clf':>10} {'C+H':>6} | {'Miss%':>6}")
print("=" * 100)
for sid in scenarios:
    r = run_trip(os.path.join(EVAL_DIR, sid), sid, clf)
    m = r['metrics']
    row = {'scenario': sid, 'candidate_miss_pct': 100 * r['candidate_miss_rate']}
    for k in M:
        row[f'{k}_f1']    = m[k]['macro_f1']
        row[f'{k}_prec']  = m[k]['precision']
        row[f'{k}_rec']   = m[k]['recall']
        row[f'{k}_top1']  = m[k]['top1_accuracy']
        row[f'{k}_cont']  = m[k]['route_continuity']
    rows.append(row)
    print(f"{sid:<10} | {row['classifier_only_f1']:>7.3f} {row['hmm_only_f1']:>7.3f} "
          f"{row['classifier_hmm_f1']:>7.3f} | {row['classifier_only_top1']:>9.3f} "
          f"{row['hmm_only_top1']:>6.3f} {row['classifier_hmm_top1']:>6.3f} | "
          f"{100*row['classifier_only_cont']:>10.1f} {100*row['classifier_hmm_cont']:>6.1f} | "
          f"{row['candidate_miss_pct']:>6.2f}")

print("=" * 100)
mean = lambda key: float(np.mean([r[key] for r in rows]))
std  = lambda key: float(np.std([r[key] for r in rows]))
print(f"\n{'Method':<18} {'F1':>14} {'Precision':>10} {'Recall':>8} {'Top-1':>8} {'Continuity':>11}")
for k in M:
    print(f"{NAMES[k]:<18} {mean(k+'_f1'):>8.3f}±{std(k+'_f1'):.3f} {mean(k+'_prec'):>10.3f} "
          f"{mean(k+'_rec'):>8.3f} {mean(k+'_top1'):>8.3f} {100*mean(k+'_cont'):>10.1f}%")
print(f"\nCandidate miss rate (mean over trips): {mean('candidate_miss_pct'):.2f}%")

with open(os.path.join(BASE_DIR, 'clf_vs_hmm_results.csv'), 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader(); w.writerows(rows)

# ── plot ──────────────────────────────────────────────────────────────────────
x = np.arange(len(scenarios)); w = 0.27
colors = {'classifier_only': 'steelblue', 'hmm_only': 'darkorange', 'classifier_hmm': 'seagreen'}
fig, ax = plt.subplots(1, 3, figsize=(19, 5))
for i, k in enumerate(M):
    ax[0].bar(x + (i - 1) * w, [r[k + '_f1'] for r in rows], w, label=NAMES[k], color=colors[k])
    ax[1].bar(x + (i - 1) * w, [r[k + '_top1'] for r in rows], w, label=NAMES[k], color=colors[k])
ax[0].set_title('Route F1 (arc-set overlap)'); ax[1].set_title('Top-1 observation accuracy')
for a in ax[:2]:
    a.set_ylim(0, 1.05); a.legend(); a.grid(axis='y', alpha=.3)
ax[2].bar(x - w / 2, [100 * r['classifier_only_cont'] for r in rows], w, label='Continuity: classifier only', color='steelblue')
ax[2].bar(x + w / 2, [100 * r['classifier_hmm_cont'] for r in rows], w, label='Continuity: classifier + HMM', color='seagreen')
ax[2].plot(x, [r['candidate_miss_pct'] for r in rows], 'o--', color='tomato', label='Candidate miss rate %')
ax[2].set_ylim(0, 110); ax[2].set_title('Route continuity & candidate miss rate (%)')
ax[2].legend(fontsize=8); ax[2].grid(axis='y', alpha=.3)
for a in ax:
    a.set_xticks(x); a.set_xticklabels(scenarios, rotation=40, ha='right', fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, 'clf_vs_hmm_comparison.png'), dpi=150)
print("\nSaved clf_vs_hmm_comparison.png, clf_vs_hmm_results.csv")
