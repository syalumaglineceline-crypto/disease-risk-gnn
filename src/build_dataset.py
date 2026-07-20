"""
Turns the three raw files into the graph the models use:
  nodes_enriched.csv  one row per district: features + a risk tier
  edges.csv           one row per pair of districts that share a border

A few notes on decisions I made here:
  - I match districts across the three files by a normalised name (lowercased,
    letters only) and fall back to a close-string match, because the map, the
    census and the COVID file all spell districts slightly differently. This
    name reconciliation is most of the work.
  - "Neighbours" means shared border. I compute it from the boundary polygons.
  - The risk tier (low/medium/high) is cases-per-100k split into thirds. It is
    the thing we try to predict. Note it is NOT fed back in as a feature.
"""
import os, json, csv, re, math, difflib
from collections import defaultdict, Counter
from shapely.geometry import shape
from shapely.strtree import STRtree

HERE = os.path.dirname(__file__)
RAW = os.path.join(HERE, "..", "data", "raw")
OUT = os.path.join(HERE, "..")

def norm(s):
    return re.sub(r"[^a-z]", "", (s or "").lower())

def approx_area_km2(geom):
    # rough equal-area: convert lon/lat to local km, then shoelace.
    # good enough for a population-density feature.
    def ring_area(coords):
        if len(coords) < 3:
            return 0.0
        lat = sum(c[1] for c in coords) / len(coords)
        k = math.cos(math.radians(lat))
        pts = [(c[0] * 111.32 * k, c[1] * 110.57) for c in coords]
        s = 0.0
        for i in range(len(pts) - 1):
            s += pts[i][0] * pts[i + 1][1] - pts[i + 1][0] * pts[i][1]
        return abs(s) / 2
    polys = [geom["coordinates"]] if geom["type"] == "Polygon" else geom["coordinates"]
    a = 0.0
    for poly in polys:
        for i, ring in enumerate(poly):
            a += ring_area(ring) if i == 0 else -ring_area(ring)
    return max(a, 1.0)

# --- load districts (nodes) ---
gj = json.load(open(os.path.join(RAW, "districts.geojson")))
feats = gj["features"]
geoms = [shape(f["geometry"]).buffer(0) for f in feats]
areas = [approx_area_km2(f["geometry"]) for f in feats]
names = [(f["properties"]["NAME_1"], f["properties"]["NAME_2"]) for f in feats]
print("districts:", len(feats))

# --- neighbour edges via a spatial index ---
tree = STRtree(geoms)
buffered = [g.buffer(0.003) for g in geoms]   # ~300m tolerance
edges = set()
for i, bg in enumerate(buffered):
    for j in tree.query(bg):
        j = int(j)
        if j > i and bg.intersects(geoms[j]):
            edges.add((i, j))
print("edges:", len(edges))

# --- census features ---
census = {}
with open(os.path.join(RAW, "census2011.csv"), encoding="utf-8", errors="ignore") as fh:
    for r in csv.DictReader(fh):
        census[norm(r["District name"])] = r
ckeys = list(census)

# --- incidence ---
inc = {}
with open(os.path.join(RAW, "district_latest.csv")) as fh:
    for r in csv.DictReader(fh):
        inc[norm(r["District"])] = r
ikeys = list(inc)

def match(name, index, keys):
    k = norm(name)
    if k in index:
        return index[k]
    hit = difflib.get_close_matches(k, keys, n=1, cutoff=0.82)
    return index[hit[0]] if hit else None

# --- build node rows ---
deg = defaultdict(int)
for i, j in edges:
    deg[i] += 1
    deg[j] += 1

rows = []
c_ok = i_ok = 0
for nid, (state, dist) in enumerate(names):
    c = match(dist, census, ckeys)
    ii = match(dist, inc, ikeys)
    if c: c_ok += 1
    if ii: i_ok += 1
    pop = int(c["Population"]) if c else 0
    log_pop = round(math.log(pop), 3) if pop else ""
    density = round(pop / areas[nid], 1) if pop else ""
    literacy = round(int(c["Literate"]) / pop * 100, 1) if c and pop else ""
    worker = round(int(c["Workers"]) / pop * 100, 1) if c and pop else ""
    conf = int(ii["Confirmed"]) if ii and ii["Confirmed"] else 0
    cp100k = round(conf / pop * 100000, 1) if pop else ""
    rows.append([nid, state, dist, log_pop, density, literacy, worker, conf, cp100k, deg[nid]])

print(f"census matched: {c_ok}/{len(names)}   incidence matched: {i_ok}/{len(names)}")

# --- risk tiers: cases-per-100k split into thirds ---
vals = sorted(r[8] for r in rows if r[8] != "")
t1, t2 = vals[len(vals) // 3], vals[2 * len(vals) // 3]
for r in rows:
    cp = r[8]
    r.append("" if cp == "" else ("high" if cp > t2 else "medium" if cp > t1 else "low"))

with open(os.path.join(OUT, "nodes_enriched.csv"), "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["node_id", "state", "district", "log_pop", "density",
                "literacy_pct", "worker_pct", "confirmed", "cases_per_100k",
                "n_neighbours", "risk_tier"])
    w.writerows(rows)

with open(os.path.join(OUT, "edges.csv"), "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["src_node_id", "dst_node_id"])
    for i, j in sorted(edges):
        w.writerow([i, j])

print("wrote nodes_enriched.csv and edges.csv")
