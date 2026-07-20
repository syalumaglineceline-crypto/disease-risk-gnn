"""
Trains and compares four ways of predicting a district's risk tier, all on the
same train/test split so the numbers are comparable:

  1. Random Forest on the district's own features only (no graph)
  2. A small neural net on the same features (no graph)
  3. A GCN: same features, but each district also sees its neighbours' features
  4. Label propagation: ignore features, spread known risk across the border graph

The whole question of the project is whether the graph helps, and if so, how.
"""
import os, csv, warnings
import numpy as np
warnings.filterwarnings("ignore")
import torch, torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import GCNConv
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from collections import defaultdict, Counter

HERE = os.path.dirname(__file__)
ROOT = os.path.join(HERE, "..")
LAB = {"low": 0, "medium": 1, "high": 2}
FEATURES = ["log_pop", "density", "literacy_pct", "worker_pct"]

def load():
    rows = list(csv.DictReader(open(os.path.join(ROOT, "nodes_enriched.csv"))))
    N = len(rows)
    X = np.zeros((N, len(FEATURES)), np.float32)
    y = np.full(N, -1)
    for r in rows:
        i = int(r["node_id"])
        if all(r[f] != "" for f in FEATURES):
            X[i] = [float(r[f]) for f in FEATURES]
        if r["risk_tier"] in LAB:
            y[i] = LAB[r["risk_tier"]]
    edges = []
    for r in csv.DictReader(open(os.path.join(ROOT, "edges.csv"))):
        edges.append((int(r["src_node_id"]), int(r["dst_node_id"])))
    return X, y, edges, N

def main():
    X, y, edges, N = load()
    labelled = [i for i in range(N) if y[i] >= 0 and X[i].any()]
    yl = y[labelled]
    tr, te = train_test_split(labelled, test_size=0.3, random_state=42, stratify=yl)

    # standardise using training stats only
    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
    Xs = (X - mu) / sd

    # 1. Random Forest, features only
    rf = RandomForestClassifier(300, random_state=42).fit(Xs[tr], y[tr])
    rf_acc = accuracy_score(y[te], rf.predict(Xs[te]))

    # graph tensors
    ei = torch.tensor([[a for a, b in edges] + [b for a, b in edges],
                       [b for a, b in edges] + [a for a, b in edges]], dtype=torch.long)
    xt = torch.tensor(Xs)
    yt = torch.tensor(y)
    trm = torch.zeros(N, dtype=torch.bool); trm[tr] = True
    tem = torch.zeros(N, dtype=torch.bool); tem[te] = True

    class MLP(torch.nn.Module):
        def __init__(s): super().__init__(); s.l1 = Linear(4, 16); s.l2 = Linear(16, 3)
        def forward(s, x, e): return s.l2(F.dropout(F.relu(s.l1(x)), 0.5, s.training))

    class GCN(torch.nn.Module):
        def __init__(s): super().__init__(); s.c1 = GCNConv(4, 16); s.c2 = GCNConv(16, 3)
        def forward(s, x, e): return s.c2(F.dropout(F.relu(s.c1(x, e)), 0.5, s.training), e)

    def train_net(Model, seeds=5):
        accs = []
        for sd in range(seeds):
            torch.manual_seed(sd)
            m = Model(); opt = torch.optim.Adam(m.parameters(), lr=0.01, weight_decay=5e-4)
            for _ in range(200):
                m.train(); opt.zero_grad()
                F.cross_entropy(m(xt, ei)[trm], yt[trm]).backward(); opt.step()
            m.eval()
            accs.append((m(xt, ei).argmax(1)[tem] == yt[tem]).float().mean().item())
        return float(np.mean(accs))

    mlp_acc = train_net(MLP)
    gcn_acc = train_net(GCN)

    # 4. label propagation on the border graph
    nb = defaultdict(list)
    for a, b in edges: nb[a].append(b); nb[b].append(a)
    P = np.zeros((N, 3))
    for i in tr: P[i, y[i]] = 1.0
    for _ in range(50):
        Pn = P.copy()
        for node in range(N):
            v = [P[x] for x in nb[node]]
            if v: Pn[node] = np.mean(v, 0)
        for i in tr: Pn[i] = 0; Pn[i, y[i]] = 1.0
        P = Pn
    lp_acc = accuracy_score(y[te], P.argmax(1)[te])

    results = {
        "Guessing": 1 / 3,
        "Features (RF)": rf_acc,
        "Neural net": mlp_acc,
        "Vanilla GCN": gcn_acc,
        "Label propagation": lp_acc,
    }
    print("\nAccuracy on the same {}-district test set:".format(len(te)))
    for k, v in results.items():
        print(f"  {k:<20}{v*100:5.1f}%")

    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        ks = list(results); vs = [results[k] * 100 for k in ks]
        colours = ["#898781" if k != "Label propagation" else "#2a78d6" for k in ks]
        plt.figure(figsize=(7, 4))
        plt.bar(ks, vs, color=colours)
        plt.ylabel("accuracy (%)"); plt.ylim(0, 65)
        plt.axhline(33.3, ls="--", c="#999", lw=1)
        plt.title("Does the graph help, and how?")
        plt.xticks(rotation=20, ha="right"); plt.tight_layout()
        plt.savefig(os.path.join(ROOT, "figures", "model_comparison.png"), dpi=130)
        print("\nsaved figures/model_comparison.png")
    except Exception as e:
        print("(skipped chart:", e, ")")

if __name__ == "__main__":
    main()
