"""
Explainability with SHAP.

I explain the "combined" model: a Random Forest that gets each district's own
features PLUS a summary of its neighbours' known risk (the share of training
neighbours that are low/medium/high). SHAP then tells us, both overall and for
a single district, which factors pushed the prediction.

This is the part that turns a prediction into a reason someone could act on.
"""
import os, csv, warnings
import numpy as np
warnings.filterwarnings("ignore")
from collections import defaultdict, Counter
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import shap

HERE = os.path.dirname(__file__)
ROOT = os.path.join(HERE, "..")
LAB = {"low": 0, "medium": 1, "high": 2}
BASE = ["log_pop", "density", "literacy_pct", "worker_pct"]
ALL = BASE + ["nbr_low%", "nbr_med%", "nbr_high%", "n_train_nbrs"]

def main():
    rows = list(csv.DictReader(open(os.path.join(ROOT, "nodes_enriched.csv"))))
    N = len(rows)
    X = np.zeros((N, 4), np.float32); y = np.full(N, -1); name = {}
    for r in rows:
        i = int(r["node_id"]); name[i] = (r["state"], r["district"])
        if all(r[f] != "" for f in BASE): X[i] = [float(r[f]) for f in BASE]
        if r["risk_tier"] in LAB: y[i] = LAB[r["risk_tier"]]
    edges = [(int(r["src_node_id"]), int(r["dst_node_id"]))
             for r in csv.DictReader(open(os.path.join(ROOT, "edges.csv")))]
    nb = defaultdict(list)
    for a, b in edges: nb[a].append(b); nb[b].append(a)

    labelled = [i for i in range(N) if y[i] >= 0 and X[i].any()]
    tr, te = train_test_split(labelled, test_size=0.3, random_state=42, stratify=y[labelled])
    trs = set(tr)

    def nbfeat(node):
        v = [y[x] for x in nb[node] if x in trs]
        c = Counter(v); n = len(v) or 1
        return [c.get(0, 0)/n, c.get(1, 0)/n, c.get(2, 0)/n, len(v)]

    Xtr = np.array([list(X[i]) + nbfeat(i) for i in tr]); ytr = y[tr]
    Xte = np.array([list(X[i]) + nbfeat(i) for i in te])
    rf = RandomForestClassifier(300, random_state=42).fit(Xtr, ytr)

    arr = np.array(shap.TreeExplainer(rf).shap_values(Xte))
    if arr.ndim == 3 and arr.shape[0] == 3:
        arr = np.transpose(arr, (1, 2, 0))
    glob = np.abs(arr).mean(axis=(0, 2)) if arr.ndim == 3 else np.abs(arr).mean(0)

    print("Overall: how much each factor drives the prediction (mean |SHAP|)")
    for n, v in sorted(zip(ALL, glob), key=lambda z: -z[1]):
        print(f"  {n:<14}{v:.4f}")

    pred = rf.predict(Xte)
    for k, node in enumerate(te):
        if pred[k] == 2 and y[node] == 2:
            st, ds = name[node]
            contr = arr[k, :, 2] if arr.ndim == 3 else arr[k]
            print(f"\nWhy {ds}, {st} was flagged high risk:")
            for n, v in sorted(zip(ALL, contr), key=lambda z: -abs(z[1]))[:5]:
                print(f"  {n:<14}{v:+.3f}  ({'toward high' if v > 0 else 'away from high'})")
            break

    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        order = np.argsort(glob)
        plt.figure(figsize=(7, 4))
        plt.barh([ALL[i] for i in order], [glob[i] for i in order], color="#2a78d6")
        plt.xlabel("mean |SHAP| (impact on prediction)")
        plt.title("What drives the disease-risk prediction")
        plt.tight_layout()
        plt.savefig(os.path.join(ROOT, "figures", "shap_importance.png"), dpi=130)
        print("\nsaved figures/shap_importance.png")
    except Exception as e:
        print("(skipped chart:", e, ")")

if __name__ == "__main__":
    main()
