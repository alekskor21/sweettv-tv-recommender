"""Stage 52: is a 0.003 difference real? (1) ranker seed variance, (2) paired bootstrap CIs over
users for the key comparisons, (3) candidate recall contributed by each generator."""
import numpy as np
import pandas as pd

from common import (P, SEED, SPLITS, apk, candidate_pool, gt_variants, interactions, load_airings,
                    load_schedule, test_uids)
from hybrid import FEATURES, FEATURES_EXT, add_labels, build_candidates, ranker, recs_from_scores
from methods import ContentBased, PersonalHistory, Popularity, PureSVD, top_k

air = load_airings()
sched = load_schedule()
sched_full = pd.read_parquet(P / "schedule.parquet")
sched_full["week"] = sched_full["start_time"].dt.isocalendar().week.astype(int)
n_users = int(air["uid"].max()) + 1
all_users = np.arange(n_users)
test = test_uids()
train_users = np.setdiff1d(all_users, test)
sp = SPLITS["val"]
gt = gt_variants(air, sp["target"], test)["A5_half"]
users = np.array(sorted(gt))

c = build_candidates(air, sched, sched_full, sp["train"], sp["target"], n_users, all_users)[0]
c = add_labels(c, air, sp["target"], all_users)
tr, te = c[c["uid"].isin(train_users)], c[c["uid"].isin(test)].copy()
fb = list(te.groupby("tv_show_id")["i_pop"].first().sort_values(ascending=False).index)


def per_user_ap(recs):
    return np.array([apk(set(gt[u]), recs[u]) for u in users])


# (1) seed variance of the re-ranker (early-stopping split is random)
ap = {}
for name, feats in [("hybrid (final)", FEATURES), ("hybrid + regularity", FEATURES_EXT)]:
    scores = []
    for seed in range(5):
        m = ranker(feats)
        m.set_params(random_state=seed)
        m.fit(tr[feats], tr["label"])
        te["p"] = m.predict_proba(te[feats])[:, 1]
        a = per_user_ap(recs_from_scores(te, "p", test, fb))
        scores.append(a.mean())
        if seed == SEED % 5:
            ap[name] = a
    print(f"{name}: MAP@5 over 5 seeds = {np.mean(scores):.4f} +- {np.std(scores):.4f} (min {min(scores):.4f}, max {max(scores):.4f})")

# single methods, same protocol
inter = interactions(air, sp["train"], "any")
pool = candidate_pool(sched, sp["target"])
cand = pool.index.values
popr = Popularity().fit(inter)
pop_list = list(popr.pop.reindex(cand).fillna(0).sort_values(ascending=False).index)
ap["popularity"] = per_user_ap(top_k(popr.score(users, cand), users, cand, pop_list))
ap["personal history"] = per_user_ap(top_k(PersonalHistory().fit(inter).score(users, cand), users, cand, pop_list))
ap["PureSVD"] = per_user_ap(top_k(PureSVD(128).fit(inter, n_users).score(users, cand), users, cand, pop_list))
cb = ContentBased(10, ("cat", "genre", "decade", "ch", "dir", "act")).fit(inter, n_users, sched_full, sp["target"][1])
ap["content-based"] = per_user_ap(top_k(cb.score(users, cand), users, cand, pop_list))

# (2) paired bootstrap over users
rng = np.random.default_rng(SEED)
B = 2000
idx = rng.integers(0, len(users), size=(B, len(users)))
print("\npaired bootstrap (2,000 resamples of the 1,246 users), difference in MAP@5 with 95% CI:")
pairs = [("hybrid (final)", "personal history"), ("hybrid (final)", "popularity"),
         ("personal history", "popularity"), ("PureSVD", "popularity"), ("PureSVD", "personal history"),
         ("content-based", "popularity"), ("hybrid + regularity", "hybrid (final)")]
rows = []
for a_name, b_name in pairs:
    d = ap[a_name] - ap[b_name]
    boots = d[idx].mean(axis=1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    rows.append((a_name, b_name, round(d.mean(), 4), round(lo, 4), round(hi, 4)))
    print(f"  {a_name} - {b_name}: {d.mean():+.4f} [{lo:+.4f}, {hi:+.4f}]")
pd.DataFrame(rows, columns=["A", "B", "diff", "ci_low", "ci_high"]).to_csv("outputs/52_bootstrap.csv", index=False)
pd.DataFrame(ap, index=users).to_parquet(P / "val_per_user_ap.parquet")

# (3) candidate recall by generator (test users, A5_half GT items)
pos = {(u, s) for u in users for s in gt[u]}
n_pos = len(pos)
hist = inter[inter["uid"].isin(users) & inter["tv_show_id"].isin(cand)]
src = {"own history": set(zip(hist["uid"], hist["tv_show_id"]))}
for name, model, n in [("PureSVD top-50", PureSVD(128).fit(inter, n_users), 50), ("content-based top-20", cb, 20)]:
    s = model.score(users, cand)
    top = np.argpartition(-np.where(np.isfinite(s), s, -1e9), n, axis=1)[:, :n]
    src[name] = {(u, cand[j]) for u, row in zip(users, top) for j in row}
src["popularity top-30"] = {(u, s) for u in users for s in pop_list[:30]}
union = set().union(*src.values())
print("\ncandidate recall of GT items:")
for name, s in src.items():
    only = s - set().union(*(v for k, v in src.items() if k != name))
    print(f"  {name}: {len(s & pos) / n_pos:.3f} (unique to this source: {len(only & pos) / n_pos:.3f})")
print(f"  union: {len(union & pos) / n_pos:.3f}")
