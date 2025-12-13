import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

def main():
    feats = np.load("../data/noise_features/noise_features.npy")
    labs = np.load("../data/noise_features/noise_labels.npy")
    print("Loaded features:", feats.shape, labs.shape)

    scaler = StandardScaler()
    X = scaler.fit_transform(feats)
    y = labs

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    aucs = []
    for train_idx, test_idx in skf.split(X, y):
        Xtr, Xte = X[train_idx], X[test_idx]
        ytr, yte = y[train_idx], y[test_idx]

        clf = LogisticRegression(max_iter=2000, solver="saga")  # L2 by default
        clf.fit(Xtr, ytr)
        probs = clf.predict_proba(Xte)[:, 1]
        auc = roc_auc_score(yte, probs)
        aucs.append(auc)
        print("Fold AUC:", auc)

    print("Mean AUC:", np.mean(aucs), "Std:", np.std(aucs))

if __name__ == "__main__":
    main()
