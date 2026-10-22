"""Stage 21: MF variants after ALS on all overlaps underperformed popularity.
(a) ALS with positives = user-programme pairs with >= T total minutes in train;
(b) PureSVD (Cremonesi et al. 2010): truncated SVD of the log1p(minutes) matrix, which predicts
    viewing intensity rather than the binary "watched at all" preference of ALS."""
import itertools
import os
import time

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import numpy as np
import pandas as pd
import scipy.sparse as sps
from scipy.sparse.linalg import svds

from common import (SEED, SPLITS, candidate_pool, evaluate, fill_to_k, gt_variants_v1 as gt_variants, interactions,
                    load_airings, load_schedule, popularity_ranking, test_uids)
from importlib import import_module

als_mod = import_module("20_als")

air = load_airings()
sched = load_schedule()
uids = test_uids()
n_users = int(air["uid"].max()) + 1
sp = SPLITS["val"]
pool = candidate_pool(sched, sp["target"])
gts = gt_variants(air, sp["target"], uids)
inter = interactions(air, sp["train"], "any")
pop = popularity_ranking(inter, pool)
rows = []

for min_minutes, factors, alpha in itertools.product([30, 60, 180], [16, 32, 64], [1.0, 10.0]):
    t1 = time.time()
    sub = inter[inter["overlap_sec"] >= min_minutes * 60]
    model, m, items = als_mod.fit_als(sub, n_users, factors, 0.1, alpha, iters=20)
    recs = als_mod.als_recs(model, m, items, uids, pool, pop)
    rows.append({"model": "ALS", "min_minutes": min_minutes, "factors": factors, "alpha": alpha,
                 **evaluate(recs, gts), "sec": round(time.time() - t1, 1)})
    print(rows[-1], flush=True)

items = pd.Index(np.sort(inter["tv_show_id"].unique()))
x = sps.csr_matrix((np.log1p(inter["overlap_sec"].values / 60).astype(np.float64),
                    (inter["uid"].values, items.get_indexer(inter["tv_show_id"]))), shape=(n_users, len(items)))
cand = np.flatnonzero(items.isin(pool.index))
for factors in [8, 16, 32, 64, 128]:
    t1 = time.time()
    u, s, vt = svds(x, k=factors, random_state=SEED)
    scores = (u[uids] * s) @ vt[:, cand]
    top = np.argsort(-scores, axis=1)[:, :5]
    recs = {uid: fill_to_k(items[cand[row]].tolist(), pop) for uid, row in zip(uids, top)}
    rows.append({"model": "PureSVD", "min_minutes": 0, "factors": factors, "alpha": None,
                 **evaluate(recs, gts), "sec": round(time.time() - t1, 1)})
    print(rows[-1], flush=True)

res = pd.DataFrame(rows)
res.to_csv("outputs/21_mf_variants.csv", index=False)
print(res.sort_values("T10sec", ascending=False).head(6).to_string(index=False))
