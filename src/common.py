"""Shared protocol for every method: windows, interactions, candidate pool,
ground truth, MAP@5 and the cold-start fallback. All methods import from here,
so the split, pool and metric are identical across methods."""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
P = ROOT / "data" / "parquet"
OUT = ROOT / "outputs"
K = 5
SEED = 42

# validation: fit on ISO weeks 11-20, evaluate on 21-30; final: fit on 11-30, predict 31-42
SPLITS = {
    "val": {"train": (11, 20), "target": (21, 30)},
    "test": {"train": (11, 30), "target": (31, 42)},
}


def load_airings():
    """Per (user, programme airing) watched seconds, labelled airtime only (tv_show_id != 0)."""
    return pd.read_parquet(P / "airings.parquet")


def load_users():
    return pd.read_parquet(P / "user_index.parquet")["user_id"]


def test_uids():
    users = load_users()
    tmpl = pd.read_parquet(P / "submission_template.parquet")
    pos = pd.Index(users).get_indexer(tmpl["user_id"])
    assert (pos >= 0).all(), "every template user must be in the view log"
    return pos


def load_schedule():
    s = pd.read_parquet(P / "schedule_dedup.parquet")
    s["week"] = s["start_time"].dt.isocalendar().week.astype(int)
    return s


def interactions(air, weeks, rule="any"):
    """User-programme pairs in long form: watched seconds and number of airings that pass `rule`.

    rule: "any" (overlap > 0), "300s" (>= 300 s per airing), "half" (>= 50% of the airing),
          "300s_or_half".
    """
    a = air[(air["week"] >= weeks[0]) & (air["week"] <= weeks[1])]
    if rule == "300s":
        a = a[a["overlap_sec"] >= 300]
    elif rule == "half":
        a = a[a["frac"] >= 0.5]
    elif rule == "300s_or_half":
        a = a[(a["overlap_sec"] >= 300) | (a["frac"] >= 0.5)]
    return a.groupby(["uid", "tv_show_id"], as_index=False).agg(
        overlap_sec=("overlap_sec", "sum"), n_airings=("overlap_sec", "size"))


def candidate_pool(sched, weeks):
    """Labelled programmes that air in the target window, with their airtime there (seconds)."""
    s = sched[(sched["tv_show_id"] != 0) & (sched["week"] >= weeks[0]) & (sched["week"] <= weeks[1])]
    return s.groupby("tv_show_id")["duration"].sum().rename("airtime_sec")


def ground_truth(air, weeks, uids, k=K):
    """Proxy of the hidden target: user's top-k programmes by watched seconds in the window."""
    t = interactions(air, weeks, "any")
    t = t[t["uid"].isin(uids)].sort_values(["uid", "overlap_sec"], ascending=[True, False])
    return t.groupby("uid")["tv_show_id"].apply(lambda s: list(s.head(k))).to_dict()


def apk(actual, predicted, k=K):
    """Average precision at k, Kaggle convention: normalise by min(len(actual), k)."""
    if not actual:
        return 0.0
    hits, score, seen = 0, 0.0, set()
    for i, p in enumerate(predicted[:k]):
        if p in actual and p not in seen:
            hits += 1
            score += hits / (i + 1)
        seen.add(p)
    return score / min(len(actual), k)


def mapk(gt, recs, k=K):
    """Mean AP@k over users that have ground truth (users without target activity are skipped)."""
    return float(np.mean([apk(set(a), recs.get(u, []), k) for u, a in gt.items()]))


def popularity_ranking(inter, pool):
    """Fallback and baseline: programmes ranked by number of distinct viewers, restricted to the pool."""
    pop = inter.groupby("tv_show_id")["uid"].nunique()
    pop = pop[pop.index.isin(pool.index)].sort_values(ascending=False)
    return list(pop.index)


def fill_to_k(ranked, fallback, k=K):
    """Take the method's ranked ids, then pad with the popularity fallback, no duplicates."""
    out = []
    for x in list(ranked) + list(fallback):
        if x not in out:
            out.append(x)
        if len(out) == k:
            break
    return out


def gt_variants(air, weeks, uids):
    """Ground-truth definitions. LB calibration (s01 popularity 0.28234, s02 personal-by-seconds
    0.27882 public) is reproduced only by count-based targets, see 13_gt_grid / 14_count_hypothesis:
    A5_half (primary): top-5 programmes by number of airings the user watched >= 50% of;
    A5_any: top-5 by number of airings with any overlap; T5sec: a-priori top-5 by watched seconds."""
    t = air[(air["week"] >= weeks[0]) & (air["week"] <= weeks[1]) & air["uid"].isin(uids)].assign(one=1)

    def top(df, col, n):
        g = df.groupby(["uid", "tv_show_id"], as_index=False)[col].sum().sort_values(["uid", col], ascending=[True, False])
        return g.groupby("uid")["tv_show_id"].apply(lambda s: list(s.head(n))).to_dict()

    return {
        "A5_half": top(t[t["frac"] >= 0.5], "one", 5),
        "A5_any": top(t, "one", 5),
        "T5sec": top(t, "overlap_sec", 5),
    }


def evaluate(recs, gts):
    return {name: round(mapk(gt, recs), 4) for name, gt in gts.items()}


def gt_variants_v1(air, weeks, uids):
    """The first set of ground-truth definitions (before LB calibration point #2), kept so that the
    exploratory scripts 20, 21, 30, 31 reproduce their saved outputs: T5sec, T10sec, A5_300."""
    t = air[(air["week"] >= weeks[0]) & (air["week"] <= weeks[1]) & air["uid"].isin(uids)].assign(one=1)

    def top(df, col, n):
        g = df.groupby(["uid", "tv_show_id"], as_index=False)[col].sum().sort_values(["uid", col], ascending=[True, False])
        return g.groupby("uid")["tv_show_id"].apply(lambda s: list(s.head(n))).to_dict()

    return {
        "T5sec": top(t, "overlap_sec", 5),
        "T10sec": top(t, "overlap_sec", 10),
        "A5_300": top(t[t["overlap_sec"] >= 300], "one", 5),
    }
