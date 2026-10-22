"""Fill the report's data tables from outputs/ (so no number is retyped by hand) and write the final HTML.

Reads report/report_src.html (all prose lives there and is never overwritten), replaces the
<!--TABLE:...--> and <!--EXAMPLE:n--> markers, writes report/korostelev_aleksandr_exam_report.html.
"""
import html
import json
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs"
SRC = ROOT / "report" / "report_src.html"
DST = ROOT / "report" / "korostelev_aleksandr_exam_report.html"
CAT_EN = {"Инфо": "Info", "Развлечения": "Entertainment", "Познавательное": "Educational", "Детям": "Kids",
          "Сериалы": "Series", "Фильмы": "Films", "Другое": "Other", "Спорт": "Sport"}


def f4(x):
    return f"{x:.4f}"


def methods_table():
    m = pd.read_csv(OUT / "40_methods_eval.csv")
    best = m.sort_values("A5_half", ascending=False).groupby("method").head(1).set_index("method")
    grid = pd.read_csv(OUT / "13_gt_grid.csv")
    g = lambda by, flt, n: grid[(grid["rank by"] == by) & (grid["airing filter"] == flt) & (grid["top-N"].astype(str) == n)].iloc[0]
    sec_half, sec_any, sec_t5 = g("airings", ">=50%", "5"), g("airings", "any", "5"), g("seconds", "any", "5")
    hyb = [l for l in (OUT / "logs" / "50_hybrid_eval.log").read_text().splitlines() if l.startswith("hybrid (GBM")][0].split()
    lb = pd.read_csv(OUT / "53_lb_calibration.csv").set_index("submission")
    lbs = lambda k: f"{lb.loc[k, 'LB public']:.5f} / {lb.loc[k, 'LB private']:.5f}"
    rows = [
        ("Popularity (reference)", "distinct viewers", best.loc["popularity"], lbs("s01 popularity")),
        ("Personal history", "own airing counts", best.loc["personal history"], "—"),
        ("Personal history, seconds", "own watched seconds", {"A5_half": sec_half["personal"], "A5_any": sec_any["personal"], "T5sec": sec_t5["personal"]}, lbs("s02 personal history (seconds)")),
        ("MF: ALS [1]", best.loc["ALS", "params"], best.loc["ALS"], "—"),
        ("MF: PureSVD [2]", best.loc["PureSVD", "params"], best.loc["PureSVD"], "—"),
        ("Content-based [3]", "min_airings=10, all fields + channel", best.loc["content-based"], "—"),
        ("<b>Hybrid (final)</b>", "candidates + re-ranker", {"A5_half": float(hyb[3]), "A5_any": float(hyb[4]), "T5sec": float(hyb[5])}, "<b>" + lbs("s03 hybrid") + "</b>"),
    ]
    body = "".join(f"<tr><td>{n}</td><td>{html.escape(str(p))}</td><td class=n>{f4(r['A5_half'])}</td><td class=n>{f4(r['A5_any'])}</td>"
                   f"<td class=n>{f4(r['T5sec'])}</td><td class=n>{l}</td></tr>" for n, p, r, l in rows)
    return ("<table><caption>Table II. Validation MAP@5 (history 11–20 → target 21–30, 1,246 users) and Kaggle score</caption>"
            "<tr><th>Method</th><th>Configuration</th><th class=n>A5_half</th><th class=n>A5_any</th><th class=n>T5sec</th>"
            "<th class=n>Kaggle public / private</th></tr>" + body + "</table>")


def calibration_table():
    grid = pd.read_csv(OUT / "13_gt_grid.csv")
    keep = [("seconds", "any", "5"), ("seconds", "any", "10"), ("airings", "any", "5"), ("airings", ">=300s", "5"),
            ("airings", ">=50%", "5"), ("seconds", "any", "all")]
    rows = ""
    for by, flt, n in keep:
        r = grid[(grid["rank by"] == by) & (grid["airing filter"] == flt) & (grid["top-N"].astype(str) == n)].iloc[0]
        name = f"top-{n} by {by}" + ("" if flt == "any" else f", airings {flt}") if n != "all" else "all programmes watched"
        bold = by == "airings" and flt == ">=50%"
        cell = lambda x: f"<b>{x}</b>" if bold else x
        rows += (f"<tr><td>{cell(html.escape(name))}</td><td class=n>{cell(f4(r['popularity']))}</td><td class=n>{cell(f4(r['personal']))}</td>"
                 f"<td class=n>{cell(f'{r.personal / r.popularity:.2f}')}</td></tr>")
    lb = pd.read_csv(OUT / "53_lb_calibration.csv").set_index("submission")
    p, s = lb.loc["s01 popularity"], lb.loc["s02 personal history (seconds)"]
    rows += (f"<tr><td><i>Kaggle (public / private)</i></td><td class=n>{p['LB public']:.5f} / {p['LB private']:.5f}</td>"
             f"<td class=n>{s['LB public']:.5f} / {s['LB private']:.5f}</td>"
             f"<td class=n>{s['LB public'] / p['LB public']:.2f} / {s['LB private'] / p['LB private']:.2f}</td></tr>")
    return ("<table><caption>Table I. Which hidden target fits the leaderboard? Validation MAP@5 of the two probes</caption>"
            "<tr><th>Ground-truth definition (weeks 21–30)</th><th class=n>Popularity</th><th class=n>Personal (seconds)</th>"
            "<th class=n>Ratio</th></tr>" + rows + "</table>")


def bootstrap_table():
    b = pd.read_csv(OUT / "52_bootstrap.csv")
    rows = "".join(f"<tr><td>{html.escape(r.A)} − {html.escape(r.B)}</td><td class=n>{r.diff:+.4f}</td>"
                   f"<td class=n>[{r.ci_low:+.4f}, {r.ci_high:+.4f}]</td></tr>" for r in b.itertuples())
    return ("<table><caption>Table III. Paired bootstrap over users (2,000 resamples), A5_half</caption>"
            "<tr><th>Comparison</th><th class=n>ΔMAP@5</th><th class=n>95% CI</th></tr>" + rows + "</table>")


def short(t, n=30):
    t = re.sub(r"^(Х/ф|Т/с|М/с|Д/с|М/ф)\s*", "", t).replace("'", "").strip(" .")
    return html.escape(t if len(t) <= n else t[: n - 1].rstrip() + "…")


def example_table(i):
    u = json.loads((OUT / "70_examples.json").read_text())[i]
    order = ["popularity", "personal history", "MF: ALS", "MF: PureSVD", "content-based", "hybrid (final)"]

    def cell(x, mark=True):
        flag = (" ✓" if x["hit"] else "") + ("" if x["seen"] else " <i>(new)</i>") if mark else ""
        txt = f"<b>{x['id']}</b>{flag}<br>{short(x['title'])}<br><span class=cat>{CAT_EN.get(x['category'], x['category'])}</span>"
        return f"<td class=hit>{txt}</td>" if mark and x["hit"] else f"<td>{txt}</td>"

    rows = "".join(f"<tr><td>{m}<br><span class=cat>AP@5 = {u['ap_at_5'][m]:.2f}</span></td>" + "".join(cell(x) for x in u["methods"][m]) + "</tr>" for m in order)
    rows += "<tr class=actual><td>Actually watched 21–30 (GT)</td>" + "".join(cell(x, False) for x in u["actual_21_30"]) + "</tr>"
    hist = "; ".join(f"{h['id']} {short(h['title'], 26)} ×{h['n_airings']}" for h in u["history_top5"])
    return (f"<table class=ex><caption>Table {['IV', 'V'][i]}. {u['label'].capitalize()}: user {u['user_id']} — "
            f"{u['airings_11_20']} airings / {u['distinct_11_20']} programmes in weeks 11–20</caption>"
            "<tr><th>Method</th><th>#1</th><th>#2</th><th>#3</th><th>#4</th><th>#5</th></tr>" + rows +
            f"<tr><td colspan=6 class=hist>Own top-5 in weeks 11–20 (airings): {hist}</td></tr></table>")


src = SRC.read_text()
src = src.replace("<!--TABLE:methods-->", methods_table()).replace("<!--TABLE:calibration-->", calibration_table())
src = src.replace("<!--TABLE:bootstrap-->", bootstrap_table())
src = src.replace("<!--EXAMPLE:0-->", example_table(0)).replace("<!--EXAMPLE:1-->", example_table(1))
assert "<!--TABLE" not in src and "<!--EXAMPLE" not in src
DST.write_text(src)
print(f"wrote {DST}")
