"""Hybrid = candidate generation + learned re-ranking.

Candidates per user: own history in the pool, PureSVD top-50, content-based top-20, popularity top-30.
Features use only the history window's views and the target window's *schedule* (what airs, where,
how often). The re-ranker (gradient boosting) is fitted on non-test users and evaluated on test users.
"""
import numpy as np
import pandas as pd
import scipy.sparse as sps

from common import P, SEED, candidate_pool, gt_variants, interactions
from methods import ContentBased, PureSVD

CB_FIELDS = ("cat", "genre", "decade", "ch", "dir", "act")
FEATURES = [
    "h_cnt", "h_cnt_half", "h_sec", "h_cnt_recent", "h_share", "h_rate", "h_expected",
    "u_cnt", "u_items", "i_pop", "i_pop_recent", "i_cnt_half", "i_air_target", "i_air_train",
    "i_air_ratio", "i_new", "i_cat", "svd", "cb", "ch_aff",
]
# regularity / recency features added after the first hybrid submission (s03)
FEATURES_EXT = FEATURES + ["h_weeks", "h_gap", "h_cnt_last2", "i_weeks_train", "i_in_last_week"]


def _top_n(scores, n):
    idx = np.argpartition(-scores, n, axis=1)[:, :n]
    return idx


def build_candidates(air, sched, sched_full, hist_w, targ_w, n_users, users, n_svd=50, n_cb=20, n_pop=30):
    """Return one row per (uid, tv_show_id) candidate with features; `users` = uids to score."""
    users = np.asarray(users)
    pool = candidate_pool(sched, targ_w)
    cand = pool.index.values
    col_of = pd.Index(cand)
    inter = interactions(air, hist_w, "any")
    half = interactions(air, hist_w, "half")
    recent = interactions(air, (hist_w[1] - 3, hist_w[1]), "any")

    svd = PureSVD(128).fit(inter, n_users)
    cb = ContentBased(10, CB_FIELDS).fit(inter, n_users, sched_full, targ_w[1])
    s_svd = svd.score(users, cand)
    s_cb = cb.score(users, cand)

    lab = sched[sched["tv_show_id"] != 0]
    tgt = lab[(lab["week"] >= targ_w[0]) & (lab["week"] <= targ_w[1])]
    trn = lab[(lab["week"] >= hist_w[0]) & (lab["week"] <= hist_w[1])]
    # airings per week, so a 12-week test target and a 10-week validation target are on one scale
    i_air_target = tgt.groupby("tv_show_id").size() / (targ_w[1] - targ_w[0] + 1)
    i_air_train = trn.groupby("tv_show_id").size() / (hist_w[1] - hist_w[0] + 1)
    i_pop = inter.groupby("tv_show_id")["uid"].nunique()
    i_pop_recent = recent.groupby("tv_show_id")["uid"].nunique()
    i_cnt_half = half.groupby("tv_show_id")["n_airings"].sum()
    cat = sched_full.drop_duplicates("tv_show_id").set_index("tv_show_id")["tv_show_category"]
    cat_codes = {c: i for i, c in enumerate(sorted(cat.dropna().unique()))}

    # channel affinity: user's share of watched airings per channel, dotted with the programme's
    # distribution of target-window airings over channels
    sd = pd.read_parquet(P / "schedule_dedup.parquet", columns=["channel_id"])
    a = air[(air["week"] >= hist_w[0]) & (air["week"] <= hist_w[1])]
    ch = sd["channel_id"].values[a["sched_idx"].values]
    chans = pd.Index(np.unique(np.r_[ch, tgt["channel_id"].values]))
    uc = sps.csr_matrix((np.ones(len(a), np.float32), (a["uid"].values, chans.get_indexer(ch))),
                        shape=(n_users, len(chans)))
    uc = sps.diags(1 / np.maximum(np.asarray(uc.sum(1)).ravel(), 1)) @ uc
    t_in = tgt[tgt["tv_show_id"].isin(cand)]
    ic = sps.csr_matrix((np.ones(len(t_in), np.float32),
                         (col_of.get_indexer(t_in["tv_show_id"]), chans.get_indexer(t_in["channel_id"]))),
                        shape=(len(cand), len(chans)))
    ic = sps.diags(1 / np.maximum(np.asarray(ic.sum(1)).ravel(), 1)) @ ic
    ch_aff = (uc[users] @ ic.T).toarray()

    # candidate generation
    row_of = pd.Index(users)
    hist = inter[inter["uid"].isin(users) & inter["tv_show_id"].isin(cand)][["uid", "tv_show_id"]]
    parts = [hist]
    for scores, n in ((s_svd, n_svd), (s_cb, n_cb)):
        idx = _top_n(np.where(np.isfinite(scores), scores, -1e9), n)
        parts.append(pd.DataFrame({"uid": np.repeat(users, n), "tv_show_id": cand[idx.ravel()]}))
    top_pop = i_pop.reindex(cand).fillna(0).sort_values(ascending=False).index[:n_pop].values
    parts.append(pd.DataFrame({"uid": np.repeat(users, n_pop), "tv_show_id": np.tile(top_pop, len(users))}))
    c = pd.concat(parts, ignore_index=True).drop_duplicates()

    # features
    key = ["uid", "tv_show_id"]
    c = c.merge(inter.rename(columns={"n_airings": "h_cnt", "overlap_sec": "h_sec"}), on=key, how="left")
    c = c.merge(half[key + ["n_airings"]].rename(columns={"n_airings": "h_cnt_half"}), on=key, how="left")
    c = c.merge(recent[key + ["n_airings"]].rename(columns={"n_airings": "h_cnt_recent"}), on=key, how="left")
    c[["h_cnt", "h_sec", "h_cnt_half", "h_cnt_recent"]] = c[["h_cnt", "h_sec", "h_cnt_half", "h_cnt_recent"]].fillna(0)
    u_tot = inter.groupby("uid")["n_airings"].sum()
    u_items = inter.groupby("uid").size()
    c["u_cnt"] = c["uid"].map(u_tot).fillna(0)
    c["u_items"] = c["uid"].map(u_items).fillna(0)
    c["h_share"] = c["h_cnt"] / c["u_cnt"].clip(lower=1)
    c["i_air_target"] = c["tv_show_id"].map(i_air_target).fillna(0)
    c["i_air_train"] = c["tv_show_id"].map(i_air_train).fillna(0)
    c["i_air_ratio"] = c["i_air_target"] / c["i_air_train"].clip(lower=1)
    c["h_rate"] = c["h_cnt"] / c["i_air_train"].clip(lower=1)
    c["h_expected"] = c["h_rate"] * c["i_air_target"]
    c["i_new"] = (c["i_air_train"] == 0).astype(int)
    c["i_pop"] = c["tv_show_id"].map(i_pop).fillna(0)
    c["i_pop_recent"] = c["tv_show_id"].map(i_pop_recent).fillna(0)
    c["i_cnt_half"] = c["tv_show_id"].map(i_cnt_half).fillna(0)
    c["i_cat"] = c["tv_show_id"].map(cat).map(cat_codes).fillna(-1).astype(int)
    r, k = row_of.get_indexer(c["uid"]), col_of.get_indexer(c["tv_show_id"])
    c["svd"] = s_svd[r, k]
    c["cb"] = s_cb[r, k]
    c.loc[~np.isfinite(c["svd"]), "svd"] = np.nan
    c.loc[~np.isfinite(c["cb"]), "cb"] = np.nan
    c["ch_aff"] = ch_aff[r, k]

    # regularity and recency of the user's viewing of the programme, and of its airing
    wk = a.groupby(["uid", "tv_show_id"]).agg(h_weeks=("week", "nunique"), h_last=("week", "max")).reset_index()
    c = c.merge(wk, on=key, how="left")
    c["h_weeks"] = c["h_weeks"].fillna(0)
    c["h_gap"] = (hist_w[1] - c["h_last"]).fillna(hist_w[1] - hist_w[0] + 1)
    c = c.drop(columns="h_last")
    last2 = interactions(air, (hist_w[1] - 1, hist_w[1]), "any")
    c = c.merge(last2[key + ["n_airings"]].rename(columns={"n_airings": "h_cnt_last2"}), on=key, how="left")
    c["h_cnt_last2"] = c["h_cnt_last2"].fillna(0)
    c["i_weeks_train"] = c["tv_show_id"].map(trn.groupby("tv_show_id")["week"].nunique()).fillna(0)
    c["i_in_last_week"] = c["tv_show_id"].isin(trn.loc[trn["week"] == hist_w[1], "tv_show_id"]).astype(int)
    return c, pool, i_pop


def add_labels(c, air, targ_w, users):
    gt = gt_variants(air, targ_w, users)["A5_half"]
    pos = {(u, s) for u, items in gt.items() for s in items}
    c["label"] = [(u, s) in pos for u, s in zip(c["uid"].values, c["tv_show_id"].values)]
    return c


def ranker(features=FEATURES, max_leaf_nodes=31, learning_rate=0.05):
    from sklearn.ensemble import HistGradientBoostingClassifier
    cat_mask = [f == "i_cat" for f in features]
    return HistGradientBoostingClassifier(max_iter=600, learning_rate=learning_rate, max_leaf_nodes=max_leaf_nodes,
                                          min_samples_leaf=50, l2_regularization=1.0,
                                          categorical_features=cat_mask, early_stopping=True,
                                          validation_fraction=0.1, random_state=SEED)


def recs_from_scores(c, score_col, users, fallback, k=5, cold_fallback=True):
    """Top-k by score. Cold users (no labelled airing in the history window, u_cnt == 0) get the
    popularity fallback, the same rule as every single method; cold_fallback=False reproduces s03,
    where the re-ranker also ordered candidates for cold users from item features alone."""
    from common import fill_to_k
    warm = c[c["u_cnt"] > 0] if cold_fallback else c
    top = warm.sort_values(["uid", score_col], ascending=[True, False]).groupby("uid")["tv_show_id"].apply(list).to_dict()
    return {u: fill_to_k(top.get(u, []), fallback, k) for u in users}
