"""Stage 20: matrix factorisation for implicit feedback (ALS, Hu-Koren-Volinsky 2008, `implicit`).

Confidence = 1 + alpha * log1p(watched minutes). Repeats are allowed (same rule for all methods),
so already-watched programmes are not filtered. Programmes absent from the training matrix
(new in the target window) cannot be scored by MF; the list is padded with popularity.
"""
import itertools
import os
import time

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import numpy as np
import pandas as pd
import scipy.sparse as sps
from implicit.als import AlternatingLeastSquares

from common import (SEED, SPLITS, candidate_pool, evaluate, fill_to_k, gt_variants_v1 as gt_variants, interactions,
                    load_airings, load_schedule, popularity_ranking, test_uids)


def fit_als(inter, n_users, factors, reg, alpha, iters):
    items = pd.Index(np.sort(inter["tv_show_id"].unique()))
    conf = 1 + alpha * np.log1p(inter["overlap_sec"].values / 60)
    m = sps.csr_matrix((conf.astype(np.float32), (inter["uid"].values, items.get_indexer(inter["tv_show_id"]))),
                       shape=(n_users, len(items)))
    model = AlternatingLeastSquares(factors=factors, regularization=reg, iterations=iters,
                                    random_state=SEED, num_threads=8)
    model.fit(m, show_progress=False)
    return model, m, items


def als_recs(model, m, items, uids, pool, fallback, k=5):
    cand = np.flatnonzero(items.isin(pool.index))
    ids, _ = model.recommend(uids, m[uids], N=k, items=cand, filter_already_liked_items=False)
    return {u: fill_to_k(items[row].tolist(), fallback, k) for u, row in zip(uids, ids)}


if __name__ == "__main__":
    t0 = time.time()
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
    grid = itertools.product([32, 64, 128], [0.01, 0.1], [1.0, 10.0, 40.0])
    for factors, reg, alpha in grid:
        t1 = time.time()
        model, m, items = fit_als(inter, n_users, factors, reg, alpha, iters=20)
        recs = als_recs(model, m, items, uids, pool, pop)
        rows.append({"factors": factors, "reg": reg, "alpha": alpha, **evaluate(recs, gts),
                     "sec": round(time.time() - t1, 1)})
        print(rows[-1], flush=True)
    res = pd.DataFrame(rows)
    res.to_csv("outputs/20_als_grid.csv", index=False)
    print(res.sort_values("T10sec", ascending=False).head(5).to_string(index=False))
    print(f"runtime {time.time() - t0:.1f}s")
