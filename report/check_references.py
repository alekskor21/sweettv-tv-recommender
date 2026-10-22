"""Reference check for the report (not part of the pipeline): Crossref metadata for every DOI and the
HTTP status of the DOI resolver and of each direct URL. Generic User-Agent, no personal data sent."""
import json
import subprocess
import urllib.request

UA = {"User-Agent": "exam-reference-check/1.0"}
DOIS = ["10.1109/ICDM.2008.22", "10.1145/1864708.1864721", "10.1007/978-0-387-85820-3_3",
        "10.1145/2959100.2959190", "10.1201/9780429246593"]
URLS = ["https://www.kaggle.com/competitions/sweettv-tv-program-recommender",
        "https://benfred.github.io/implicit/",
        "https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingClassifier.html",
        "https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.svds.html"]


def status(url, follow):
    args = ["curl", "-s", "-o", "/dev/null", "-A", UA["User-Agent"], "--max-time", "30", "-w", "%{http_code} %{url_effective}"]
    if follow:
        args.append("-L")
    return subprocess.run(args + [url], capture_output=True, text=True).stdout


for doi in DOIS:
    try:
        req = urllib.request.Request(f"https://api.crossref.org/works/{doi}", headers=UA)
        m = json.load(urllib.request.urlopen(req, timeout=30))["message"]
        people = m.get("author") or m.get("editor") or []
        names = ", ".join(f"{a.get('given', '')} {a.get('family', '')}".strip() for a in people)
        year = (m.get("published-print") or m.get("published") or m.get("issued"))["date-parts"][0][0]
        print(f"DOI {doi}\n  crossref 200 | {m.get('title', [''])[0]}\n  people: {names}\n"
              f"  venue: {(m.get('container-title') or [''])[0]} | publisher {m.get('publisher')} | year {year} | pp {m.get('page')}")
    except Exception as e:
        print(f"DOI {doi}\n  crossref FAILED {e}")
    print(f"  doi.org (no redirect): {status('https://doi.org/' + doi, False)}")
for u in URLS:
    print(f"URL {u}: {status(u, True)}")
