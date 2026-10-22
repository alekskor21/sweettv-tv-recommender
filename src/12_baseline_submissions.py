"""Stage 12: refit the two baselines on weeks 11-30 and write test submissions (pool = weeks 31-42)."""
import pandas as pd

from common import P, SPLITS, candidate_pool, fill_to_k, interactions, load_airings, load_schedule, popularity_ranking, test_uids
from submission import write_submission

air = load_airings()
sched = load_schedule()
uids = test_uids()
sp = SPLITS["test"]
pool = candidate_pool(sched, sp["target"])
inter = interactions(air, sp["train"], "any")
pop = popularity_ranking(inter, pool)
print("test pool size:", len(pool), "| popularity top-5:", pop[:5])

write_submission({u: pop[:5] for u in uids}, "s01_popularity", pool.index)

own = inter[inter["tv_show_id"].isin(pool.index)].sort_values(["uid", "overlap_sec"], ascending=[True, False])
own_lists = own.groupby("uid")["tv_show_id"].apply(list).to_dict()
recs = {u: fill_to_k(own_lists.get(u, []), pop) for u in uids}
print("test users needing popularity fill:", sum(len(own_lists.get(u, [])) < 5 for u in uids))
write_submission(recs, "s02_personal_history", pool.index)

full = pd.read_parquet(P / "schedule.parquet")
top = full[full["tv_show_id"].isin(pop[:5])].drop_duplicates("tv_show_id").set_index("tv_show_id")
print(top.loc[pop[:5], ["tv_show_title", "tv_show_category", "channel_title"]].to_string())
