"""Stage 55: context only - where would our public scores sit on the frozen 2020 public leaderboard?
Input: `kaggle competitions leaderboard sweettv-tv-program-recommender --download -p data/leaderboard`
(kept out of git: it lists participants' names). Prints aggregates only."""
from pathlib import Path

import pandas as pd

csv = next(Path("data/leaderboard").glob("*publicleaderboard*.csv"))
s = pd.read_csv(csv, encoding="utf-8-sig")["Score"].sort_values(ascending=False).reset_index(drop=True)
print(f"source: {csv.name} | teams: {len(s)} | best {s.iloc[0]:.5f} | median {s.median():.5f}")
for name, x in [("s01 popularity", 0.28234), ("s02 personal history", 0.27882), ("s03 hybrid", 0.42177)]:
    above = int((s > x).sum())
    print(f"{name} {x:.5f}: {above} teams above -> rank {above + 1} of {len(s) + 1}")
