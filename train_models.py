"""
train_models.py  –  Train and compare 8 models for PathSnap candidate scoring
==============================================================================
Models:
  1. Logistic Regression
  2. Decision Tree
  3. K-Nearest Neighbours
  4. SVM (RBF)
  5. Gradient Boosting (sklearn)   <-- required
  6. Random Forest
  7. XGBoost
  8. LightGBM

Metrics: CV Macro F1, Test Macro F1, Top-1 Accuracy, Log Loss, Time
"""

import os, sys, time, joblib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.linear_model  import LogisticRegression
from sklearn.ensemble      import (RandomForestClassifier,
                                   GradientBoostingClassifier)
from sklearn.tree          import DecisionTreeClassifier
from sklearn.neighbors     import KNeighborsClassifier
from sklearn.svm           import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline      import Pipeline
from sklearn.model_selection import GroupKFold, cross_validate
from sklearn.metrics       import (f1_score, log_loss,
                                   classification_report,
                                   confusion_matrix, ConfusionMatrixDisplay)
import xgboost  as xgb
import lightgbm as lgb

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from features import build_dataset, FEATURE_NAMES

base_dir  = os.path.dirname(os.path.abspath(__file__))
workspace = os.path.dirname(base_dir)
train_dir = os.path.join(workspace, 'Student folder', 'Student folder', 'train')
stud_dir  = os.path.join(workspace, 'Student folder', 'Student folder', 'student_test')
out_dir   = os.path.join(base_dir, 'models')
os.makedirs(out_dir, exist_ok=True)

# ── load data ─────────────────────────────────────────────────────────────────
# Use first 20 trips for speed; increase MAX_TRAIN_TRIPS if time allows
MAX_TRAIN_TRIPS = 20
print(f"Loading training data ({MAX_TRAIN_TRIPS} trips)...")
X_train, y_train, g_train, _ = build_dataset(train_dir, max_trips=MAX_TRAIN_TRIPS,
                                              verbose=True)
print(f"\nTrain: {X_train.shape[0]} candidates, "
      f"{y_train.sum()} positive ({100*y_train.mean():.1f}%)")

print("\nLoading student-test data (10 trips)...")
X_test, y_test, g_test, _ = build_dataset(stud_dir, verbose=False)
print(f"Test:  {X_test.shape[0]} candidates")

# ── helpers ───────────────────────────────────────────────────────────────────
gkf = GroupKFold(n_splits=3)   # 3-fold for speed; use 5 if time allows

def cv_scores(clf, X, y, groups):
    cv = cross_validate(clf, X, y, groups=groups,
                        cv=gkf, scoring=['f1_macro'],
                        return_train_score=False, n_jobs=-1)
    return float(cv['test_f1_macro'].mean())

def top1_accuracy(clf, X_t, y_t, g_t):
    if hasattr(clf, 'predict_proba'):
        scores = clf.predict_proba(X_t)[:, 1]
    else:
        scores = clf.decision_function(X_t)
    obs_ids = np.unique(g_t)
    correct = sum(
        1 for oid in obs_ids
        if y_t[g_t == oid][np.argmax(scores[g_t == oid])] == 1
    )
    return correct / max(len(obs_ids), 1)

# ── model definitions ─────────────────────────────────────────────────────────
models = {
    'Logistic Regression': Pipeline([
        ('sc',  StandardScaler()),
        ('clf', LogisticRegression(max_iter=500, C=1.0, random_state=42))
    ]),
    'Decision Tree': Pipeline([
        ('sc',  StandardScaler()),
        ('clf', DecisionTreeClassifier(max_depth=8, random_state=42))
    ]),
    'K-Nearest Neighbours': Pipeline([
        ('sc',  StandardScaler()),
        ('clf', KNeighborsClassifier(n_neighbors=7, n_jobs=-1))
    ]),
    'SVM (RBF)': Pipeline([
        ('sc',  StandardScaler()),
        ('clf', SVC(kernel='rbf', C=1.0, probability=True, random_state=42,
                    max_iter=500))
    ]),
    'Gradient Boosting': GradientBoostingClassifier(
        n_estimators=200, max_depth=5, learning_rate=0.1,
        subsample=0.8, random_state=42),
    'Random Forest': RandomForestClassifier(
        n_estimators=200, max_depth=12, n_jobs=-1, random_state=42),
    'XGBoost': xgb.XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8,
        eval_metric='logloss', random_state=42, n_jobs=-1),
    'LightGBM': lgb.LGBMClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8,
        random_state=42, n_jobs=-1, verbose=-1),
}

# ── train & evaluate ──────────────────────────────────────────────────────────
results = {}

print("\n" + "="*78)
print(f"{'Model':<22} {'CV F1':>8} {'Test F1':>8} {'Top1':>8} {'LogLoss':>9} {'t(s)':>6}")
print("="*78)

for name, clf in models.items():
    t0 = time.time()
    cv_f1   = cv_scores(clf, X_train, y_train, g_train)
    clf.fit(X_train, y_train)
    y_pred  = clf.predict(X_test)
    t_f1    = f1_score(y_test, y_pred, average='macro', zero_division=0)
    top1    = top1_accuracy(clf, X_test, y_test, g_test)
    if hasattr(clf, 'predict_proba'):
        ll = log_loss(y_test, clf.predict_proba(X_test))
    else:
        ll = float('nan')
    dt = time.time() - t0

    results[name] = dict(cv_f1=cv_f1, test_f1=t_f1, top1=top1,
                         log_loss=ll, time=dt, clf=clf)
    ll_str = f"{ll:.4f}" if not np.isnan(ll) else "  N/A "
    print(f"{name:<22} {cv_f1:>8.4f} {t_f1:>8.4f} {top1:>8.4f} "
          f"{ll_str:>9} {dt:>6.1f}")

print("="*78)

# ── pick best ─────────────────────────────────────────────────────────────────
best_name = max(results, key=lambda n: results[n]['cv_f1'])
best_clf  = results[best_name]['clf']
print(f"\nBest model (CV F1): {best_name}  "
      f"({results[best_name]['cv_f1']:.4f})")

joblib.dump(best_clf, os.path.join(out_dir, 'best_model.joblib'))
for name, r in results.items():
    fname = (name.lower()
             .replace(' ', '_')
             .replace('(', '').replace(')', '')) + '.joblib'
    joblib.dump(r['clf'], os.path.join(out_dir, fname))
print(f"Models saved to {out_dir}/")

# ── plots ─────────────────────────────────────────────────────────────────────
names  = list(results.keys())
cv_f1s = [results[n]['cv_f1']   for n in names]
t_f1s  = [results[n]['test_f1'] for n in names]
top1s  = [results[n]['top1']    for n in names]
lls    = [results[n]['log_loss'] if not np.isnan(results[n]['log_loss']) else 0
          for n in names]
times  = [results[n]['time']    for n in names]
x      = np.arange(len(names)); w = 0.3

# 1 ── F1 comparison
fig, ax = plt.subplots(figsize=(13, 5))
bars1 = ax.bar(x - w/2, cv_f1s, w, label='CV Macro F1',   color='steelblue')
bars2 = ax.bar(x + w/2, t_f1s,  w, label='Test Macro F1', color='darkorange')
ax.set_xticks(x); ax.set_xticklabels(names, rotation=35, ha='right', fontsize=9)
ax.set_ylim(0, 1.05); ax.set_ylabel('F1 Score')
ax.set_title('Model Comparison – Macro F1 Score (CV vs Student-Test)')
ax.legend(); ax.grid(axis='y', alpha=0.3)
# highlight best
ax.bar(x[names.index(best_name)] - w/2, cv_f1s[names.index(best_name)],
       w, color='navy', label=f'Best ({best_name})')
plt.tight_layout()
plt.savefig(os.path.join(base_dir, 'model_comparison_f1.png'), dpi=150)
plt.close()

# 2 ── Top-1 Accuracy + Log Loss
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].bar(names, top1s, color='green', alpha=0.8)
axes[0].set_xticklabels(names, rotation=35, ha='right', fontsize=9)
axes[0].set_ylim(0, 1.05); axes[0].set_ylabel('Top-1 Accuracy')
axes[0].set_title('Top-1 Accuracy per Model')
axes[0].grid(axis='y', alpha=0.3)

axes[1].bar(names, lls, color='tomato', alpha=0.8)
axes[1].set_xticklabels(names, rotation=35, ha='right', fontsize=9)
axes[1].set_ylabel('Log Loss (lower = better)')
axes[1].set_title('Log Loss per Model')
axes[1].grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(base_dir, 'model_comparison_top1_logloss.png'), dpi=150)
plt.close()

# 3 ── Confusion matrix for best model
y_pred_best = best_clf.predict(X_test)
cm = confusion_matrix(y_test, y_pred_best)
fig, ax = plt.subplots(figsize=(5, 4))
ConfusionMatrixDisplay(cm, display_labels=['Non-match', 'Match']).plot(
    ax=ax, cmap='Blues', colorbar=False)
ax.set_title(f'Confusion Matrix – {best_name}')
plt.tight_layout()
plt.savefig(os.path.join(base_dir, 'confusion_matrix_best.png'), dpi=150)
plt.close()

# 4 ── Feature importance (tree-based best model)
if hasattr(best_clf, 'feature_importances_'):
    imp = best_clf.feature_importances_
elif hasattr(best_clf, 'named_steps'):
    inner = best_clf.named_steps.get('clf', None)
    imp = inner.feature_importances_ if hasattr(inner, 'feature_importances_') else None
else:
    imp = None

if imp is not None:
    idx = np.argsort(imp)[::-1]
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(range(len(imp)), imp[idx], color='steelblue')
    ax.set_xticks(range(len(imp)))
    ax.set_xticklabels([FEATURE_NAMES[i] for i in idx], rotation=45, ha='right', fontsize=8)
    ax.set_title(f'Feature Importance – {best_name}')
    ax.set_ylabel('Importance'); ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(base_dir, 'feature_importance.png'), dpi=150)
    plt.close()
    print("Saved: feature_importance.png")

# 5 ── Gradient Boosting training curve (deviance per stage)
gb_name = 'Gradient Boosting'
if gb_name in results:
    gb_clf = results[gb_name]['clf']
    if hasattr(gb_clf, 'train_score_'):
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(gb_clf.train_score_, label='Train deviance', color='steelblue')
        ax.set_xlabel('Boosting iteration'); ax.set_ylabel('Deviance')
        ax.set_title('Gradient Boosting – Training Deviance per Stage')
        ax.legend(); ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(base_dir, 'gb_training_curve.png'), dpi=150)
        plt.close()
        print("Saved: gb_training_curve.png")

print("\nPlots saved:")
print("  model_comparison_f1.png")
print("  model_comparison_top1_logloss.png")
print("  confusion_matrix_best.png")

print(f"\n--- Classification Report: {best_name} ---")
print(classification_report(y_test, y_pred_best,
                             target_names=['Non-match', 'Match']))
