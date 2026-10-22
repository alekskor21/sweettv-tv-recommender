"""Stage 30: content-based filtering.

Item vector: TF-IDF over tokens built from schedule metadata of the programme id
(category, genres 1-3, directors, first 5 actors, decade of production; optionally channel).
User profile: sum of item vectors weighted by log1p(watched minutes), L2-normalised.
Score: cosine(profile, item). Repeats allowed (same rule as all methods). Unlike MF, CB can
score programmes that are new in the target window, since their metadata is in the schedule.
"""
import time

import numpy as np
import pandas as pd
import scipy.sparse as sps
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from common import (P, SPLITS, candidate_pool, evaluate, fill_to_k, gt_variants_v1 as gt_variants, interactions,
                    load_airings, load_schedule, popularity_ranking, test_uids)


def split_names(x, limit=None):
    if not isinstance(x, str):
        return []
    names = [n.strip() for n in x.split(",") if n.strip()]
    return names[:limit] if limit else names


def item_tokens(sched_full, max_week, with_channel):
    """Token list per programme id, using schedule rows up to ISO week `max_week` only."""
    s = sched_full[(sched_full["tv_show_id"] != 0) & (sched_full["week"] <= max_week)]
    toks = {}
    for row in s.drop_duplicates(["tv_show_id", "channel_id", "tv_show_genre_1", "director", "actors"]).itertuples():
        t = toks.setdefault(row.tv_show_id, set())
        t.add(f"cat={row.tv_show_category}")
        for g in (row.tv_show_genre_1, row.tv_show_genre_2, row.tv_show_genre_3):
            if isinstance(g, str):
                t.add(f"genre={g}")
        for d in split_names(row.director):
            t.add(f"dir={d}")
        for a in split_names(row.actors, 5):
            t.add(f"act={a}")
        if isinstance(row.year_of_production, str) and row.year_of_production[:4].isdigit():
            t.add(f"decade={int(row.year_of_production[:4]) // 10 * 10}")
        if with_channel:
            t.add(f"ch={row.channel_id}")
    return toks


def build_item_matrix(toks):
    ids = pd.Index(sorted(toks))
    vec = TfidfVectorizer(analyzer=lambda x: x, min_df=2, sublinear_tf=True)
    mat = vec.fit_transform([sorted(toks[i]) for i in ids])
    return ids, normalize(mat), vec


def cb_scores(inter, uids, ids, mat, n_users):
    w = np.log1p(inter["overlap_sec"].values / 60)
    col = ids.get_indexer(inter["tv_show_id"])
    ok = col >= 0
    ui = sps.csr_matrix((w[ok], (inter["uid"].values[ok], col[ok])), shape=(n_users, len(ids)))
    prof = normalize(ui[uids] @ mat)
    return prof, mat


def cb_recs(inter, uids, ids, mat, pool, fallback, n_users, k=5):
    prof, mat = cb_scores(inter, uids, ids, mat, n_users)
    cand = np.flatnonzero(ids.isin(pool.index))
    scores = (prof @ mat[cand].T).toarray()
    top = np.argsort(-scores, axis=1)[:, :k]
    return {u: fill_to_k(ids[cand[row]].tolist(), fallback, k) for u, row in zip(uids, top)}


if __name__ == "__main__":
    t0 = time.time()
    air = load_airings()
    sched = load_schedule()
    sched_full = pd.read_parquet(P / "schedule.parquet")
    sched_full["week"] = sched_full["start_time"].dt.isocalendar().week.astype(int)
    uids = test_uids()
    n_users = int(air["uid"].max()) + 1
    sp = SPLITS["val"]
    pool = candidate_pool(sched, sp["target"])
    gts = gt_variants(air, sp["target"], uids)
    inter = interactions(air, sp["train"], "any")
    pop = popularity_ranking(inter, pool)

    rows = []
    for with_channel in (False, True):
        for min_minutes in (0, 30, 180):
            t1 = time.time()
            toks = item_tokens(sched_full, sp["target"][1], with_channel)
            ids, mat, vec = build_item_matrix(toks)
            sub = inter[inter["overlap_sec"] >= min_minutes * 60]
            recs = cb_recs(sub, uids, ids, mat, pool, pop, n_users)
            rows.append({"channel token": with_channel, "profile min minutes": min_minutes,
                         "vocab": len(vec.vocabulary_), **evaluate(recs, gts), "sec": round(time.time() - t1, 1)})
            print(rows[-1], flush=True)
    res = pd.DataFrame(rows)
    res.to_csv("outputs/30_content.csv", index=False)
    print(f"runtime {time.time() - t0:.1f}s")
