"""Stage 53: leaderboard vs validation. LB scores are copied from `kaggle competitions submissions`
(outputs/kaggle_submissions.txt); validation scores come from the saved stage outputs."""
import pandas as pd

lb = {"s01 popularity": (0.28234, 0.27809), "s02 personal history (seconds)": (0.27882, 0.24850),
      "s03 hybrid (no cold rule)": (0.42177, 0.41517), "s04 hybrid (final)": (0.42152, 0.41453)}
grid = pd.read_csv("outputs/13_gt_grid.csv")
row = grid[(grid["rank by"] == "airings") & (grid["airing filter"] == ">=50%") & (grid["top-N"].astype(str) == "5")].iloc[0]
log50 = open("outputs/logs/50_hybrid_eval.log").read().splitlines()
hyb = [l for l in log50 if l.startswith("hybrid (GBM re-ranker)")][0].split()
legacy = [l for l in log50 if l.startswith("hybrid, s03 legacy")][0].split()
val = {"s01 popularity": row["popularity"], "s02 personal history (seconds)": row["personal"],
       "s03 hybrid (no cold rule)": float(legacy[-3]), "s04 hybrid (final)": float(hyb[3])}
res = pd.DataFrame([{"submission": k, "LB public": v[0], "LB private": v[1], "val A5_half": val[k],
                     "public / val": round(v[0] / val[k], 3), "private / val": round(v[1] / val[k], 3)} for k, v in lb.items()])
print(res.to_string(index=False))
grid["personal/pop"] = (grid["personal"] / grid["popularity"]).round(3)
print("\nLB ratio s02/s01: public", round(lb["s02 personal history (seconds)"][0] / lb["s01 popularity"][0], 3),
      "| private", round(lb["s02 personal history (seconds)"][1] / lb["s01 popularity"][1], 3))
res.to_csv("outputs/53_lb_calibration.csv", index=False)
