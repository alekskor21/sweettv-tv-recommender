"""Stage 31: which metadata fields help CB? Drop token groups and vary the profile threshold."""
import importlib

import pandas as pd

from common import (P, SPLITS, candidate_pool, evaluate, gt_variants_v1 as gt_variants, interactions, load_airings,
                    load_schedule, popularity_ranking, test_uids)

cb = importlib.import_module("30_content")
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
full_toks = cb.item_tokens(sched_full, sp["target"][1], with_channel=True)

feature_sets = {
    "cat+genre": ("cat=", "genre="),
    "cat+genre+decade": ("cat=", "genre=", "decade="),
    "cat+genre+channel": ("cat=", "genre=", "ch="),
    "cat+genre+decade+channel": ("cat=", "genre=", "decade=", "ch="),
    "all (incl. director, actors)": ("cat=", "genre=", "decade=", "ch=", "dir=", "act="),
}
rows = []
for name, prefixes in feature_sets.items():
    toks = {i: {t for t in ts if t.startswith(prefixes)} for i, ts in full_toks.items()}
    ids, mat, vec = cb.build_item_matrix(toks)
    for min_minutes in (180, 600):
        sub = inter[inter["overlap_sec"] >= min_minutes * 60]
        recs = cb.cb_recs(sub, uids, ids, mat, pool, pop, n_users)
        rows.append({"features": name, "profile min minutes": min_minutes, **evaluate(recs, gts)})
        print(rows[-1], flush=True)
pd.DataFrame(rows).to_csv("outputs/31_content_ablation.csv", index=False)
