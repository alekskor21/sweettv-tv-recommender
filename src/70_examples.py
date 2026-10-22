"""Stage 70: two example users, every method's Top-5 side by side (validation split: history 11-20,
actual viewing 21-30). Users are picked by rule, not by hand: the test users closest to the 90th
(heavy) and 10th (light) percentile of watched airings in weeks 11-20, among users with a GT list."""
import json

import numpy as np
import pandas as pd

from common import P, SPLITS, apk, candidate_pool, gt_variants, interactions, load_airings, load_schedule, load_users, test_uids
from hybrid import FEATURES, add_labels, build_candidates, ranker, recs_from_scores
from methods import ALS, ContentBased, PersonalHistory, Popularity, PureSVD, top_k

air = load_airings()
sched = load_schedule()
sched_full = pd.read_parquet(P / "schedule.parquet")
sched_full["week"] = sched_full["start_time"].dt.isocalendar().week.astype(int)
n_users = int(air["uid"].max()) + 1
all_users = np.arange(n_users)
test = test_uids()
sp = SPLITS["val"]
gt = gt_variants(air, sp["target"], test)["A5_half"]
inter = interactions(air, sp["train"], "any")
pool = candidate_pool(sched, sp["target"])
cand = pool.index.values

act = inter[inter["uid"].isin(list(gt))].groupby("uid")["n_airings"].sum()
picked = {}
for label, q in [("heavy viewer (p90)", 0.9), ("light viewer (p10)", 0.1)]:
    picked[label] = int((act - act.quantile(q)).abs().idxmin())
users = np.array(list(picked.values()))

popm = Popularity().fit(inter)
fb = list(popm.pop.reindex(cand).fillna(0).sort_values(ascending=False).index)
lists = {
    "popularity": top_k(popm.score(users, cand), users, cand, fb),
    "personal history": top_k(PersonalHistory().fit(inter).score(users, cand), users, cand, fb),
    "MF: PureSVD": top_k(PureSVD(128).fit(inter, n_users).score(users, cand), users, cand, fb),
    "MF: ALS": top_k(ALS(16, 0.1, 1.0, 3).fit(inter, n_users).score(users, cand), users, cand, fb),
    "content-based": top_k(ContentBased(10, ("cat", "genre", "decade", "ch", "dir", "act"))
                           .fit(inter, n_users, sched_full, sp["target"][1]).score(users, cand), users, cand, fb),
}
c = build_candidates(air, sched, sched_full, sp["train"], sp["target"], n_users, all_users)[0]
c = add_labels(c, air, sp["target"], all_users)
tr = c[~c["uid"].isin(test)]
model = ranker(FEATURES).fit(tr[FEATURES], tr["label"])
te = c[c["uid"].isin(users)].copy()
te["p"] = model.predict_proba(te[FEATURES])[:, 1]
lists["hybrid (final)"] = recs_from_scores(te, "p", users, fb)

lab = sched_full[sched_full["tv_show_id"] != 0]
title = lab.groupby("tv_show_id")["tv_show_title"].agg(lambda s: s.value_counts().index[0])
category = lab.groupby("tv_show_id")["tv_show_category"].first()
user_hash = load_users()
hist_cnt = inter.set_index(["uid", "tv_show_id"])["n_airings"]

out = []
for label, u in picked.items():
    seen = set(inter.loc[inter["uid"] == u, "tv_show_id"])
    out.append(f"\n### {label}: user_id {user_hash[u]}")
    out.append(f"weeks 11-20: {int(act[u])} airings watched, {len(seen)} distinct programmes\n")
    own = inter[inter["uid"] == u].sort_values("n_airings", ascending=False).head(5)
    out.append("Own top-5 in weeks 11-20 (history): " + "; ".join(
        f"{i} {title.get(i, '?')} ({category.get(i, '?')}, {n})" for i, n in zip(own["tv_show_id"], own["n_airings"])))
    out.append("Actually watched, weeks 21-30 (GT, A5_half): " + "; ".join(
        f"{i} {title.get(i, '?')} ({category.get(i, '?')})" for i in gt[u]) + "\n")
    out.append("| # | " + " | ".join(lists) + " |")
    out.append("|---|" + "---|" * len(lists))
    for r in range(5):
        cells = []
        for name in lists:
            i = lists[name][u][r]
            mark = ("**hit** " if i in gt[u] else "") + ("seen" if i in seen else "new")
            cells.append(f"{i} {title.get(i, '?')[:38]} ({category.get(i, '?')}; {mark})")
        out.append(f"| {r + 1} | " + " | ".join(cells) + " |")
    hits = {name: sum(i in gt[u] for i in lists[name][u]) for name in lists}
    out.append("\nhits out of 5: " + ", ".join(f"{k} {v}" for k, v in hits.items()))
    out.append("AP@5: " + ", ".join(f"{k} {apk(set(gt[u]), lists[k][u]):.4f}" for k in lists))
text = "\n".join(out)
(P.parent.parent / "outputs" / "70_examples.md").write_text(text)
print(text)

# machine-readable copy for the report tables
export = []
for label, u in picked.items():
    seen = set(inter.loc[inter["uid"] == u, "tv_show_id"])
    item = lambda i: {"id": int(i), "title": title.get(i, "?"), "category": category.get(i, "?")}
    export.append({
        "label": label, "user_id": user_hash[u], "airings_11_20": int(act[u]), "distinct_11_20": len(seen),
        "history_top5": [{**item(i), "n_airings": int(n)} for i, n in zip(*inter[inter["uid"] == u]
                         .sort_values("n_airings", ascending=False).head(5)[["tv_show_id", "n_airings"]].values.T)],
        "actual_21_30": [item(i) for i in gt[u]],
        "methods": {name: [{**item(i), "hit": bool(i in gt[u]), "seen": bool(i in seen)} for i in lists[name][u]]
                    for name in lists},
        "ap_at_5": {name: round(apk(set(gt[u]), lists[name][u]), 4) for name in lists},
    })
(P.parent.parent / "outputs" / "70_examples.json").write_text(json.dumps(export, ensure_ascii=False, indent=1))
