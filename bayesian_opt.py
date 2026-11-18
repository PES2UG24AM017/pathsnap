"""
bayesian_opt.py  –  Bayesian hyperparameter optimisation for best model (XGBoost)
==================================================================================
Uses Optuna to search over XGBoost hyperparameters with 5-fold GroupKFold CV
on the 80 training trips. Compares tuned vs untuned model on student-test.
"""

import os, sys, time, joblib
import numpy as np
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.metrics import f1_score, log_loss, accuracy_score
import xgboost as xgb

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from features import build_dataset, FEATURE_NAMES
from train_models import top1_accuracy

base_dir  = os.path.dirname(os.path.abspath(__file__))
workspace = os.path.dirname(base_dir)
train_dir = os.path.join(workspace, 'Student folder', 'Student folder', 'train')
stud_dir  = os.path.join(workspace, 'Student folder', 'Student folder', 'student_test')
out_dir   = os.path.join(base_dir, 'models')
os.makedirs(out_dir, exist_ok=True)

print("Loading data...")
X_tr, y_tr, g_tr, _ = build_dataset(train_dir, verbose=False)
X_te, y_te, g_te, _ = build_dataset(stud_dir,  verbose=False)
print(f"Train: {X_tr.shape}  Test: {X_te.shape}")

gkf = GroupKFold(n_splits=5)

# ── untuned baseline ─────────────────────────────────────────────────────────
untuned = xgb.XGBClassifier(
    n_estimators=300, max_depth=6, learning_rate=0.1,
    subsample=0.8, colsample_bytree=0.8,
    eval_metric='logloss', random_state=42, n_jobs=-1)

t0 = time.time()
cv_base = cross_val_score(untuned, X_tr, y_tr, groups=g_tr,
                          cv=gkf, scoring='f1_macro', n_jobs=-1)
untuned.fit(X_tr, y_tr)
base_time = time.time() - t0
y_pred_base   = untuned.predict(X_te)
base_test_f1  = f1_score(y_te, y_pred_base, average='macro', zero_division=0)
base_top1     = top1_accuracy(untuned, X_te, y_te, g_te)
base_ll       = log_loss(y_te, untuned.predict_proba(X_te))
print(f"\nUntuned XGBoost:  CV F1={cv_base.mean():.4f}  "
      f"Test F1={base_test_f1:.4f}  Top1={base_top1:.4f}  "
      f"LogLoss={base_ll:.4f}  t={base_time:.1f}s")

# ── Optuna study ──────────────────────────────────────────────────────────────
N_TRIALS = 30
print(f"\nRunning Bayesian optimisation ({N_TRIALS} trials)...")

trial_f1s = []

def objective(trial):
    params = {
        'n_estimators'   : trial.suggest_int('n_estimators', 100, 500),
        'max_depth'      : trial.suggest_int('max_depth', 3, 10),
        'learning_rate'  : trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'subsample'      : trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'gamma'          : trial.suggest_float('gamma', 0.0, 5.0),
        'reg_alpha'      : trial.suggest_float('reg_alpha', 0.0, 2.0),
        'reg_lambda'     : trial.suggest_float('reg_lambda', 0.5, 3.0),
        'eval_metric'    : 'logloss',
        'random_state'   : 42,
        'n_jobs'         : -1,
    }
    clf = xgb.XGBClassifier(**params)
    cv  = cross_val_score(clf, X_tr, y_tr, groups=g_tr,
                          cv=gkf, scoring='f1_macro', n_jobs=-1)
    score = cv.mean()
    trial_f1s.append(score)
    return score

study = optuna.create_study(direction='maximize',
                            sampler=optuna.samplers.TPESampler(seed=42))
study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False)

best_params = study.best_params
print(f"\nBest params (trial {study.best_trial.number}):")
for k, v in best_params.items():
    print(f"  {k}: {v}")
print(f"  Best CV F1: {study.best_value:.4f}")

# ── tuned model ───────────────────────────────────────────────────────────────
best_params.update({'eval_metric': 'logloss', 'random_state': 42, 'n_jobs': -1})
tuned = xgb.XGBClassifier(**best_params)

t0 = time.time()
cv_tuned = cross_val_score(tuned, X_tr, y_tr, groups=g_tr,
                           cv=gkf, scoring='f1_macro', n_jobs=-1)
tuned.fit(X_tr, y_tr)
tuned_time = time.time() - t0

y_pred_tuned  = tuned.predict(X_te)
tuned_test_f1 = f1_score(y_te, y_pred_tuned, average='macro', zero_division=0)
tuned_top1    = top1_accuracy(tuned, X_te, y_te, g_te)
tuned_ll      = log_loss(y_te, tuned.predict_proba(X_te))
print(f"\nTuned XGBoost:    CV F1={cv_tuned.mean():.4f}  "
      f"Test F1={tuned_test_f1:.4f}  Top1={tuned_top1:.4f}  "
      f"LogLoss={tuned_ll:.4f}  t={tuned_time:.1f}s")

# save tuned model
joblib.dump(tuned, os.path.join(out_dir, 'xgb_tuned.joblib'))
print("Saved → models/xgb_tuned.joblib")

# ── comparison table ──────────────────────────────────────────────────────────
print("\n" + "="*60)
print(f"{'Metric':<20} {'Untuned':>10} {'Tuned':>10}")
print("="*60)
print(f"{'CV Macro F1':<20} {cv_base.mean():>10.4f} {cv_tuned.mean():>10.4f}")
print(f"{'Test Macro F1':<20} {base_test_f1:>10.4f} {tuned_test_f1:>10.4f}")
print(f"{'Top-1 Accuracy':<20} {base_top1:>10.4f} {tuned_top1:>10.4f}")
print(f"{'Log Loss':<20} {base_ll:>10.4f} {tuned_ll:>10.4f}")
print(f"{'Inference time (s)':<20} {base_time:>10.1f} {tuned_time:>10.1f}")
print("="*60)

# ── plots ─────────────────────────────────────────────────────────────────────
# 1. Optuna optimisation history
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(trial_f1s, marker='o', ms=4, lw=1.2, color='steelblue', label='Trial F1')
running_best = np.maximum.accumulate(trial_f1s)
ax.plot(running_best, lw=2, color='red', label='Best so far')
ax.axhline(cv_base.mean(), ls='--', color='grey', label=f'Untuned baseline ({cv_base.mean():.4f})')
ax.set_xlabel('Trial'); ax.set_ylabel('CV Macro F1')
ax.set_title('Bayesian Optimisation – XGBoost (Optuna TPE)')
ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(base_dir, 'bayesian_opt_history.png'), dpi=150)
plt.close()

# 2. Untuned vs Tuned comparison bar chart
metrics   = ['CV Macro F1', 'Test Macro F1', 'Top-1 Accuracy']
untuned_v = [cv_base.mean(), base_test_f1, base_top1]
tuned_v   = [cv_tuned.mean(), tuned_test_f1, tuned_top1]
x = np.arange(len(metrics)); w = 0.3
fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(x-w/2, untuned_v, w, label='Untuned', color='steelblue')
ax.bar(x+w/2, tuned_v,   w, label='Tuned',   color='darkorange')
ax.set_xticks(x); ax.set_xticklabels(metrics)
ax.set_ylim(0, 1.05); ax.set_ylabel('Score')
ax.set_title('XGBoost: Untuned vs Bayesian-Tuned')
ax.legend(); ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(base_dir, 'bayesian_opt_comparison.png'), dpi=150)
plt.close()

print("\nPlots saved:")
print("  bayesian_opt_history.png")
print("  bayesian_opt_comparison.png")
