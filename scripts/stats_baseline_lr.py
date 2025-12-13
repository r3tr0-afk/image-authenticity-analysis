import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

X = np.load("../data/stats_features/X_stats.npy")
y = np.load("../data/stats_features/y.npy")

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

aucs = []

for train_idx, val_idx in skf.split(X, y):
    clf = LogisticRegression(
        max_iter=2000,
        n_jobs=-1,
        class_weight="balanced"
    )
    clf.fit(X[train_idx], y[train_idx])
    probs = clf.predict_proba(X[val_idx])[:, 1]
    auc = roc_auc_score(y[val_idx], probs)
    aucs.append(auc)
    print("Fold AUC:", auc)

print(f"Mean AUC: {np.mean(aucs):.4f}  Std: {np.std(aucs):.4f}")
