"""Stage 4: distribution of view-programme overlaps on labelled airtime, to choose
the "watched" threshold (absolute seconds vs fraction of programme duration)."""
from pathlib import Path

import numpy as np
import pandas as pd

P = Path(__file__).resolve().parent.parent / "data" / "parquet"
j = pd.read_parquet(P / "joined.parquet")
lab = j[j["tv_show_id"] != 0].copy()
lab["frac"] = lab["overlap_sec"] / lab["prog_sec"]

print("labelled view-programme rows:", f"{len(lab):,}")
print("overlap_sec quantiles:", lab["overlap_sec"].quantile([.1, .25, .5, .75, .9]).to_dict())
print("prog_sec quantiles (labelled airings watched):", lab["prog_sec"].quantile([.1, .5, .9]).to_dict())
hist, edges = np.histogram(lab["frac"].clip(0, 1), bins=10, range=(0, 1))
print("frac histogram (share of rows per decile of programme covered):")
print("  ", {f"{edges[i]:.1f}-{edges[i+1]:.1f}": round(h / len(lab), 3) for i, h in enumerate(hist)})

# one airing can be covered by several views of the same user (channel hopping back);
# aggregate per (user, airing) first
air = lab.groupby(["uid", "sched_idx"], as_index=False).agg(
    overlap_sec=("overlap_sec", "sum"), prog_sec=("prog_sec", "first"),
    tv_show_id=("tv_show_id", "first"), week=("week", "first"))
air["frac"] = (air["overlap_sec"] / air["prog_sec"]).clip(upper=1)
hist, _ = np.histogram(air["frac"], bins=10, range=(0, 1))
print("\nper (user, airing): rows", f"{len(air):,}")
print("  frac histogram:", {f"{edges[i]:.1f}-{edges[i+1]:.1f}": round(h / len(air), 3) for i, h in enumerate(hist)})
print("  share of total overlap time in airings with frac>=0.5:",
      round(air.loc[air["frac"] >= .5, "overlap_sec"].sum() / air["overlap_sec"].sum(), 3))

rules = {
    "any overlap": air["overlap_sec"] > 0,
    ">=300 s": air["overlap_sec"] >= 300,
    ">=600 s": air["overlap_sec"] >= 600,
    "frac>=0.25": air["frac"] >= .25,
    "frac>=0.5": air["frac"] >= .5,
    ">=300 s or frac>=0.5": (air["overlap_sec"] >= 300) | (air["frac"] >= .5),
}
print("\nrule -> airings kept, distinct (user, show) pairs, users with >=1 pair, short-programme (<600 s) share of kept")
for name, m in rules.items():
    k = air[m]
    print(f"  {name:22s} {m.mean():.3f}  {k[['uid','tv_show_id']].drop_duplicates().shape[0]:>8,}  "
          f"{k['uid'].nunique():>5,}  {(k['prog_sec'] < 600).mean():.3f}")
air.to_parquet(P / "airings.parquet", index=False)
