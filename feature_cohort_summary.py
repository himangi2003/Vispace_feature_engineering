#!/usr/bin/env python3
r"""Describe a feature set in the whole cohort (no group comparison).

Works for tumor_core, tils_tsr, immune_proximity and necrosis_feature:

    python feature_cohort_summary.py --raw all_groups_tumor_core.csv \
        --engineered all_groups_tumor_core_engineered.csv --output-dir cohort_summary

    python feature_cohort_summary.py --raw all_groups_tils_tsr.csv \
        --engineered all_groups_tils_tsr_engineered.csv \
        --tissue-file all_groups_tumor_core.csv --output-dir cohort_summary

--raw is the feature table (one row per subject or WSI); --engineered is its
output from the matching *_feature_engineering.py (same rows). Tissue amount
(analysed area, tumor region count) is read from --raw, or from --tissue-file
when the feature file has no such columns (use the tumor_core CSV; rows are
matched on group, subject_id and wsi_name).

Writes to --output-dir, each file prefixed with the feature name (e.g. tils_tsr_):

  1. reference_table.csv    per raw feature: category, units, n, missing,
                            mean, SD, P5, P25, median, P75, P95, min, max
  2. pca_variance.csv       eigenvalue, % variance, cumulative % per component
     pca_loadings.csv       loading of every engineered feature on each component
     pca_scores.csv         ID columns + PC1..PCk for every row
     pca_model.json         means, SDs and loadings to project new data onto
                            the same axes (score = sum(loading * (x - mean) / SD))
  4. tissue_dependence.csv  per engineered feature: Spearman rho with log
                            analysed area (all rows and adequate-tissue rows
                            only), median in small- vs adequate-tissue rows
  <name>_cohort_summary.html   the same, as an interactive page

Small tissue = analysed area < --min-area (default 10 mm^2) or tumor regions
< --min-regions (default 3); the columns are --area-column / --region-column.
PCA uses the correlation matrix of the continuous engineered features (0/1
columns such as flags and one-hot categories are left out); rows with a blank
engineered value are left out of the PCA. Components are kept up to
--variance (default 80%) cumulative variance, at least 2.
Standard library only.
"""
import argparse
import csv
import json
import math
import statistics as st
import sys
from pathlib import Path

IDS = ["group", "subject_id", "sample_id", "wsi_name", "n_wsi"]
MISSING = {"", "na", "nan", "none", "null"}
PATIENT = {"sex", "gender", "age", "stage", "os_months", "os_status", "os_event", "os_days",
           "ajcc_pathologic_tumor_stage", "vital_status"}
CATEGORIES = {
    "Counts": ["tumor_region_count", "tumor_island_count"],
    "Size": ["analyzed_area_mm2", "tumor_area_mm2", "tumor_boundary_length_mm"],
    "Composition": ["tumor_area_fraction"],
    "Fragmentation": ["tumor_edge_per_area_per_mm", "tumor_islands_per_mm2",
                      "tumor_largest_island_share"],
    "Shape": ["tumor_island_roundness", "tumor_island_solidity", "tumor_island_elongation",
              "tumor_edge_irregularity"],
    "Spacing": ["tumor_island_spacing_um", "tumor_island_gap_um"],
    "Tumor-stroma ratio": ["tsr_tumor_to_stroma_ratio"],
    "TIL density": ["til_stromal_pct", "til_intratumoral_pct", "til_between_tumor_pct"],
    "TIL amount": ["til_total_area_mm2"],
    "TIL location": ["til_share_inside_tumor", "til_share_touching_tumor"],
    "TIL proximity": ["til_within_50um_pct", "til_within_100um_pct", "til_within_200um_pct",
                      "til_outside_distance_median_um"],
    "Necrosis extent": ["necrosis_tissue_fraction", "necrosis_area_mm2", "necrosis_boundary_length_mm",
                        "region_count"],
}
CATEGORY_OF = {f: c for c, fs in CATEGORIES.items() for f in fs}


def number(v):
    if v is None or v.strip().lower() in MISSING:
        return None
    try:
        x = float(v)
    except ValueError:
        return None
    return x if math.isfinite(x) else None


def units(name):
    n = name.lower()
    if n.endswith("_pct") or "_pct_" in n:
        return "%"
    for suffix, unit in (("per_mm2", "1/mm²"), ("per_mm", "1/mm"), ("_mm2", "mm²"), ("_mm", "mm"),
                         ("_um2", "µm²"), ("_um", "µm")):
        if n.endswith(suffix):
            return unit
    if "fraction" in n or "share" in n:
        return "0–1"
    if n.endswith("_count") or n.startswith("n_"):
        return "count"
    if "ratio" in n:
        return "ratio"
    return "unitless"


def percentile(values, q):
    values = sorted(values)
    pos = (len(values) - 1) * q
    lo, hi = math.floor(pos), math.ceil(pos)
    return values[lo] + (values[hi] - values[lo]) * (pos - lo)


def ranks(x):
    order = sorted(range(len(x)), key=lambda i: x[i])
    r, i = [0.0] * len(x), 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and x[order[j + 1]] == x[order[i]]:
            j += 1
        for k in range(i, j + 1):
            r[order[k]] = (i + j) / 2 + 1
        i = j + 1
    return r


def spearman(a, b):
    pairs = [(x, y) for x, y in zip(a, b) if x is not None and y is not None]
    if len(pairs) < 3:
        return None, len(pairs)
    ra, rb = ranks([p[0] for p in pairs]), ranks([p[1] for p in pairs])
    ma, mb = st.fmean(ra), st.fmean(rb)
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    den = math.sqrt(sum((x - ma) ** 2 for x in ra) * sum((y - mb) ** 2 for y in rb))
    return (num / den if den else None), len(pairs)


def jacobi(matrix):
    """Eigenvalues and eigenvectors of a symmetric matrix, largest first."""
    p = len(matrix)
    a = [row[:] for row in matrix]
    v = [[float(i == j) for j in range(p)] for i in range(p)]
    for _ in range(200):
        off, i, j = max((abs(a[i][j]), i, j) for i in range(p) for j in range(i + 1, p)) if p > 1 else (0, 0, 0)
        if off < 1e-12:
            break
        theta = 0.5 * math.atan2(2 * a[i][j], a[j][j] - a[i][i])
        c, s = math.cos(theta), math.sin(theta)
        for k in range(p):
            a[i][k], a[j][k] = c * a[i][k] - s * a[j][k], s * a[i][k] + c * a[j][k]
        for k in range(p):
            a[k][i], a[k][j] = c * a[k][i] - s * a[k][j], s * a[k][i] + c * a[k][j]
        for k in range(p):
            v[k][i], v[k][j] = c * v[k][i] - s * v[k][j], s * v[k][i] + c * v[k][j]
    pairs = sorted(((a[i][i], [v[k][i] for k in range(p)]) for i in range(p)), reverse=True)
    out = []
    for value, vector in pairs:
        # Sign convention: the largest-magnitude loading is positive.
        if vector[max(range(p), key=lambda k: abs(vector[k]))] < 0:
            vector = [-x for x in vector]
        out.append((max(value, 0.0), vector))
    return out


def read(path):
    with open(path, newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        header = [c.strip() for c in reader.fieldnames or []]
        rows = [{k.strip(): (v or "") for k, v in r.items() if k is not None} for r in reader]
    if not rows:
        sys.exit(f"{path}: no data rows")
    return header, rows


def key(row):
    return (row.get("group", ""), row.get("subject_id", ""), row.get("wsi_name", ""))


def write_csv(path, header, rows):
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def fmt(x):
    return "" if x is None else repr(round(x, 6))


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw", required=True, type=Path, help="Feature CSV (tumor_core, tils_tsr, ...)")
    parser.add_argument("--engineered", required=True, type=Path,
                        help="Output of the matching *_feature_engineering.py for the same rows")
    parser.add_argument("--tissue-file", type=Path,
                        help="CSV with the tissue-amount columns (default: --raw), e.g. the tumor_core CSV")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--area-column", default="analyzed_area_mm2")
    parser.add_argument("--region-column", default="tumor_region_count")
    parser.add_argument("--min-area", type=float, default=10.0, help="mm² (default 10)")
    parser.add_argument("--min-regions", type=float, default=3, help="default 3")
    parser.add_argument("--variance", type=float, default=0.80,
                        help="Keep components up to this cumulative variance (default 0.80)")
    args = parser.parse_args()

    raw_header, raw_rows = read(args.raw)
    eng_header, eng_rows = read(args.engineered)
    tissue_path = args.tissue_file or args.raw
    tissue_header, tissue_rows = (raw_header, raw_rows) if tissue_path == args.raw else read(tissue_path)
    for column in (args.area_column, args.region_column):
        if column not in tissue_header:
            sys.exit(f"{tissue_path}: no {column!r} column - pass --tissue-file (e.g. the tumor_core "
                     "CSV) or set --area-column / --region-column")
    name = args.raw.stem[len("all_groups_"):] if args.raw.stem.startswith("all_groups_") else args.raw.stem
    raw_by_key = {key(r): r for r in raw_rows}
    if len(raw_by_key) != len(raw_rows):
        sys.exit(f"{args.raw}: duplicate group/subject_id/wsi_name rows")
    unmatched = [key(r) for r in eng_rows if key(r) not in raw_by_key]
    if unmatched:
        sys.exit(f"{args.engineered}: {len(unmatched)} rows not in --raw, e.g. {unmatched[0]}")

    def is_feature(c, rows):
        if c in IDS or c.lower() in PATIENT or c.endswith("_missing"):
            return False
        values = [r[c] for r in rows if r[c].strip().lower() not in MISSING]
        return bool(values) and all(number(v) is not None for v in values)
    def is_binary(c, rows):
        return {r[c].strip() for r in rows} <= {"0", "1", ""}
    raw_features = [c for c in raw_header if is_feature(c, raw_rows)]
    eng_features = [c for c in eng_header if is_feature(c, eng_rows) and not is_binary(c, eng_rows)]
    if len(eng_features) < 2:
        sys.exit(f"{args.engineered}: PCA needs at least 2 continuous engineered features, found {eng_features}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Reference table
    reference = []
    for f in raw_features:
        v = [x for x in (number(r[f]) for r in raw_rows) if x is not None]
        reference.append(dict(feature=f, category=CATEGORY_OF.get(f, "Other"), units=units(f),
                              n=len(v), missing=len(raw_rows) - len(v), mean=st.fmean(v),
                              sd=st.stdev(v) if len(v) > 1 else 0.0,
                              p5=percentile(v, .05), p25=percentile(v, .25), median=percentile(v, .5),
                              p75=percentile(v, .75), p95=percentile(v, .95), min=min(v), max=max(v)))
    order = list(CATEGORIES) + ["Other"]
    reference.sort(key=lambda r: (order.index(r["category"]), raw_features.index(r["feature"])))
    cols = ["feature", "category", "units", "n", "missing", "mean", "sd", "p5", "p25", "median",
            "p75", "p95", "min", "max"]
    write_csv(args.output_dir / f"{name}_reference_table.csv", cols,
              [[r[c] if c in ("feature", "category", "units", "n", "missing") else fmt(r[c]) for c in cols]
               for r in reference])

    # 2. PCA on the correlation matrix of the engineered features
    complete = [r for r in eng_rows if all(number(r[f]) is not None for f in eng_features)]
    X = [[number(r[f]) for f in eng_features] for r in complete]
    means = [st.fmean(col) for col in zip(*X)]
    sds = [st.pstdev(col) or 1.0 for col in zip(*X)]
    Z = [[(x - m) / s for x, m, s in zip(row, means, sds)] for row in X]
    n, p = len(Z), len(eng_features)
    corr = [[sum(Z[k][i] * Z[k][j] for k in range(n)) / n for j in range(p)] for i in range(p)]
    eig = jacobi(corr)
    total = sum(e for e, _ in eig) or 1.0
    cumulative, keep = 0.0, 0
    variance = []
    for i, (e, _) in enumerate(eig):
        cumulative += e / total
        variance.append(dict(component=f"PC{i + 1}", eigenvalue=e, percent=100 * e / total,
                             cumulative=100 * cumulative))
        if keep == 0 and cumulative >= args.variance:
            keep = i + 1
    keep = min(p, max(2, keep or p))
    write_csv(args.output_dir / f"{name}_pca_variance.csv", ["component", "eigenvalue", "percent", "cumulative"],
              [[v["component"], fmt(v["eigenvalue"]), fmt(v["percent"]), fmt(v["cumulative"])] for v in variance])
    loadings = [[eig[c][1][i] for c in range(keep)] for i in range(p)]
    write_csv(args.output_dir / f"{name}_pca_loadings.csv", ["feature"] + [f"PC{c + 1}" for c in range(keep)],
              [[f] + [fmt(x) for x in loadings[i]] for i, f in enumerate(eng_features)])
    id_cols = [c for c in IDS if c in eng_header]
    scores = [[sum(z[i] * eig[c][1][i] for i in range(p)) for c in range(keep)] for z in Z]
    write_csv(args.output_dir / f"{name}_pca_scores.csv", id_cols + [f"PC{c + 1}" for c in range(keep)],
              [[r[c] for c in id_cols] + [fmt(x) for x in s] for r, s in zip(complete, scores)])
    (args.output_dir / f"{name}_pca_model.json").write_text(json.dumps(dict(
        features=eng_features, means=means, sds=sds,
        loadings={f"PC{c + 1}": [eig[c][1][i] for i in range(p)] for c in range(keep)},
        percent_variance=[v["percent"] for v in variance[:keep]]), indent=2))

    # 4. Dependence on tissue amount
    area = {key(r): number(r[args.area_column]) for r in tissue_rows}
    regions = {key(r): number(r[args.region_column]) for r in tissue_rows}
    no_tissue = sum(1 for r in eng_rows if area.get(key(r)) is None)
    if no_tissue:
        print(f"Note: {no_tissue} rows have no {args.area_column} in {tissue_path}; left out of the tissue check")
    small = {k for k in area if (area[k] is not None and area[k] < args.min_area)
             or (regions[k] is not None and regions[k] < args.min_regions)}
    log_area = [math.log(area[key(r)]) if area.get(key(r)) and area[key(r)] > 0 else None for r in eng_rows]
    adequate_idx = [i for i, r in enumerate(eng_rows) if key(r) not in small]
    dependence = []
    for f in eng_features:
        values = [number(r[f]) for r in eng_rows]
        rho_all, n_all = spearman(values, log_area)
        rho_ok, n_ok = spearman([values[i] for i in adequate_idx], [log_area[i] for i in adequate_idx])
        small_vals = [v for r, v in zip(eng_rows, values) if key(r) in small and v is not None]
        ok_vals = [v for r, v in zip(eng_rows, values) if key(r) not in small and v is not None]
        dependence.append(dict(feature=f, rho_all=rho_all, n_all=n_all, rho_adequate=rho_ok, n_adequate=n_ok,
                               median_small=st.median(small_vals) if small_vals else None,
                               median_adequate=st.median(ok_vals) if ok_vals else None))
    write_csv(args.output_dir / f"{name}_tissue_dependence.csv",
              ["feature", "rho_log_area_all", "n_all", "rho_log_area_adequate", "n_adequate",
               "median_small_tissue", "median_adequate_tissue"],
              [[d["feature"], fmt(d["rho_all"]), d["n_all"], fmt(d["rho_adequate"]), d["n_adequate"],
                fmt(d["median_small"]), fmt(d["median_adequate"])] for d in dependence])
    n_small_area = sum(1 for k in area if area[k] is not None and area[k] < args.min_area)
    n_small_regions = sum(1 for k in regions if regions[k] is not None and regions[k] < args.min_regions)

    # Page
    payload = dict(
        title=name, n=len(raw_rows), tissue_source=tissue_path.name, reference=reference, variance=variance, keep=keep,
        eng_features=eng_features, loadings=loadings,
        scatter=[[s[0], s[1], key(r) in small, r.get("subject_id", "")] for r, s in zip(complete, scores)],
        dependence=dependence,
        tissue=[[log_area[i], [number(r[f]) for f in eng_features], key(r) in small]
                for i, r in enumerate(eng_rows) if log_area[i] is not None],
        rule=dict(min_area=args.min_area, min_regions=args.min_regions, n_small=len(small),
                  n_small_area=n_small_area, n_small_regions=n_small_regions,
                  area_column=args.area_column, region_column=args.region_column),
        pca_n=n)
    html = PAGE.replace("__DATA__", json.dumps(payload).replace("</", "<\\/"))
    (args.output_dir / f"{name}_cohort_summary.html").write_text(html, encoding="utf-8")

    print(f"{args.raw.name}: {len(raw_rows)} rows, {len(raw_features)} raw / {len(eng_features)} engineered features")
    print("PCA (" + str(n) + " complete rows): " + "; ".join(
        f"{v['component']} {v['percent']:.1f}% (cum {v['cumulative']:.1f}%)" for v in variance[:keep]))
    for c in range(keep):
        top = sorted(zip(loadings, eng_features), key=lambda t: -abs(t[0][c]))[:3]
        print(f"  PC{c + 1}: " + ", ".join(f"{f} {l[c]:+.2f}" for l, f in top))
    print(f"Small tissue: {len(small)} of {len(raw_rows)} rows ({n_small_area} with {args.area_column} < "
          f"{args.min_area:g}, {n_small_regions} with {args.region_column} < {args.min_regions:g})")
    print("Tissue dependence (|rho| with log area >= 0.3): " + (", ".join(
        f"{d['feature']} {d['rho_all']:+.2f}" for d in dependence
        if d["rho_all"] is not None and abs(d["rho_all"]) >= 0.3) or "none"))
    print(f"Wrote {name}_reference_table.csv, _pca_variance.csv, _pca_loadings.csv, _pca_scores.csv, "
          f"_pca_model.json, _tissue_dependence.csv and {name}_cohort_summary.html to {args.output_dir}")


PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Feature Cohort Summary</title>
<style>
:root{color-scheme:light;--page:#f9f9f7;--surface:#fcfcfb;--ink:#0b0b0b;--ink-2:#52514e;--muted:#898781;--grid:#e1e0d9;
  --axis:#c3c2b7;--ring:rgba(11,11,11,.10);--s1:#2a78d6;--s2:#eb6834;--pos:#e34948;--neg:#2a78d6;--mid:#f0efec}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){color-scheme:dark;--page:#0d0d0d;--surface:#1a1a19;--ink:#f0efec;
  --ink-2:#c3c2b7;--muted:#898781;--grid:#2c2c2a;--axis:#383835;--ring:rgba(255,255,255,.10);--s1:#3987e5;--s2:#d95926;
  --pos:#e66767;--neg:#3987e5;--mid:#383835}}
:root[data-theme="dark"]{color-scheme:dark;--page:#0d0d0d;--surface:#1a1a19;--ink:#f0efec;--ink-2:#c3c2b7;--muted:#898781;
  --grid:#2c2c2a;--axis:#383835;--ring:rgba(255,255,255,.10);--s1:#3987e5;--s2:#d95926;--pos:#e66767;--neg:#3987e5;--mid:#383835}
*{box-sizing:border-box}
body{margin:0;background:var(--page);color:var(--ink);font:14px/1.45 system-ui,-apple-system,"Segoe UI",sans-serif}
main{max-width:1180px;margin:0 auto;padding:24px 16px 48px}
h1{font-size:22px;margin:0 0 4px}h2{font-size:17px;margin:32px 0 4px}
.sub{color:var(--ink-2);margin:0 0 14px}
.panel{background:var(--surface);border:1px solid var(--ring);border-radius:10px;padding:12px;overflow-x:auto}
.two{display:grid;grid-template-columns:repeat(auto-fit,minmax(340px,1fr));gap:12px}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(260px,1fr));gap:12px}
@media (max-width:420px){.two,.grid{grid-template-columns:1fr}}
.card{background:var(--surface);border:1px solid var(--ring);border-radius:10px;padding:10px 10px 6px;min-width:0}
.card h3{font-size:13px;margin:0;overflow-wrap:anywhere}
.meta{color:var(--ink-2);font-size:12px;font-variant-numeric:tabular-nums}
table{border-collapse:collapse;width:100%;font-size:12px;font-variant-numeric:tabular-nums}
th,td{padding:5px 8px;border-bottom:1px solid var(--grid);text-align:right;white-space:nowrap}
th:first-child,td:first-child,th:nth-child(2),td:nth-child(2){text-align:left}
tr.cat td{font-weight:600;color:var(--ink);background:var(--mid);text-align:left}
svg{display:block;width:100%;height:auto}
.tick{fill:var(--muted);font-size:10px;font-variant-numeric:tabular-nums}.lab{fill:var(--ink-2);font-size:11px}
.legend{display:flex;gap:16px;flex-wrap:wrap;color:var(--ink-2);margin:0 0 8px;font-size:13px}
.legend span{display:inline-flex;align-items:center;gap:6px}.dot{width:9px;height:9px;border-radius:50%;display:inline-block}
.tip{position:fixed;pointer-events:none;background:var(--surface);color:var(--ink);border:1px solid var(--ring);border-radius:8px;
  padding:6px 9px;font-size:12px;box-shadow:0 4px 14px rgba(0,0,0,.12);display:none;z-index:5;white-space:nowrap;font-variant-numeric:tabular-nums}
</style></head><body><main>
<h1 id="h1"></h1><p class="sub" id="subtitle"></p>
<h2>1. Reference table</h2><p class="sub">Raw features before transformation; P5–P95 is the central 90% of the cohort.</p>
<div class="panel" id="ref"></div>
<h2>2. Main axes of variation (PCA of the engineered features)</h2><p class="sub" id="pcasub"></p>
<div class="two"><div class="panel"><div class="meta" style="margin-bottom:6px">Variance explained per component</div><div id="scree"></div></div>
<div class="panel"><div class="legend" id="scleg"></div><div id="scatter"></div></div></div>
<div class="panel" style="margin-top:12px"><div class="meta" style="margin-bottom:6px">Loadings (red = feature rises with the component, blue = falls); values shown where |loading| ≥ 0.3</div><div id="load"></div></div>
<h2>4. Dependence on tissue amount</h2><p class="sub" id="tissub"></p>
<div class="legend" id="tisleg"></div><div class="grid" id="tissue"></div>
</main><div class="tip" id="tip"></div>
<script>
const D = __DATA__;
const tip = document.getElementById("tip");
const fmt = v => { if (v === null || v === undefined) return "–"; const a = Math.abs(v);
  return a === 0 ? "0" : a >= 1000 ? v.toFixed(0) : a >= 10 ? v.toFixed(1) : a >= 0.1 ? v.toFixed(3) : v.toPrecision(3); };
function showTip(e, h){ tip.innerHTML = h; tip.style.display = "block";
  tip.style.left = Math.min(e.clientX + 14, innerWidth - tip.offsetWidth - 8) + "px";
  tip.style.top = Math.min(e.clientY + 14, innerHeight - tip.offsetHeight - 8) + "px"; }
function hide(){ tip.style.display = "none"; }
const R = D.rule;
document.getElementById("h1").textContent = `${D.title.replace(/_/g, " ")} features in the whole cohort`;
document.title = `${D.title.replace(/_/g, " ")} cohort summary`;
document.getElementById("subtitle").textContent = `${D.n} rows · all groups pooled · tissue amount from ${D.tissue_source}`;
// 1. reference table
let t = `<table><thead><tr><th>Feature</th><th>Units</th><th>n</th><th>Missing</th><th>Mean ± SD</th><th>P5</th><th>P25</th><th>Median</th><th>P75</th><th>P95</th></tr></thead><tbody>`, cat = "";
D.reference.forEach(r => { if (r.category !== cat) { cat = r.category; t += `<tr class="cat"><td colspan="10">${cat}</td></tr>`; }
  t += `<tr><td>${r.feature}</td><td>${r.units}</td><td>${r.n}</td><td>${r.missing}</td><td>${fmt(r.mean)} ± ${fmt(r.sd)}</td><td>${fmt(r.p5)}</td><td>${fmt(r.p25)}</td><td>${fmt(r.median)}</td><td>${fmt(r.p75)}</td><td>${fmt(r.p95)}</td></tr>`; });
document.getElementById("ref").innerHTML = t + `</tbody></table>`;
// 2a. scree: bars for % per component, cumulative as text
(function(){ const v = D.variance, W = 520, H = 300, L = 34, B = 26, T = 18, pw = W - L - 8, ph = H - T - B, bw = pw / v.length;
  const top = Math.ceil(v[0].percent / 10) * 10, Y = x => T + ph - x / top * ph;
  let s = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Variance explained">`;
  [0, top / 2, top].forEach(g => s += `<line x1="${L}" x2="${W - 8}" y1="${Y(g)}" y2="${Y(g)}" stroke="var(--grid)"/><text class="tick" x="${L - 4}" y="${Y(g) + 3}" text-anchor="end">${g}%</text>`);
  v.forEach((c, i) => { const x = L + i * bw + 2, h = Y(0) - Y(c.percent), kept = i < D.keep;
    s += `<path d="M${x},${Y(0)}V${Y(c.percent) + 4}q0,-4 4,-4h${bw - 12}q4,0 4,4V${Y(0)}Z" fill="${kept ? "var(--s1)" : "var(--axis)"}"><title>${c.component}: ${c.percent.toFixed(1)}% (cumulative ${c.cumulative.toFixed(1)}%)</title></path>`;
    s += `<text class="tick" x="${x + (bw - 4) / 2}" y="${H - 10}" text-anchor="middle">${c.component}</text>`;
    if (kept) s += `<text class="lab" x="${x + (bw - 4) / 2}" y="${Y(c.percent) - 4}" text-anchor="middle">${c.percent.toFixed(0)}%</text>`; });
  document.getElementById("scree").innerHTML = s + `</svg>`;
  document.getElementById("pcasub").textContent = `${D.keep} components explain ${v[D.keep - 1].cumulative.toFixed(0)}% of the variance (${D.pca_n} rows with complete engineered features). Kept components in blue.`; })();
// 2b. PC1 vs PC2 scatter, colored by tissue adequacy
const series = [["adequate tissue", "var(--s1)"], [`small tissue (< ${R.min_area} mm² or < ${R.min_regions} regions)`, "var(--s2)"]];
const legend = series.map(([n, c]) => `<span><i class="dot" style="background:${c}"></i>${n}</span>`).join("");
document.getElementById("scleg").innerHTML = legend; document.getElementById("tisleg").innerHTML = legend;
function scatter(pts, W, H, xl, yl, aria, xtick = fmt){ const L = 40, B = 26, T = 8, Rm = 8, pw = W - L - Rm, ph = H - T - B;
  const xs = pts.map(p => p[0]), ys = pts.map(p => p[1]);
  const x0 = Math.min(...xs), x1 = Math.max(...xs), y0 = Math.min(...ys), y1 = Math.max(...ys);
  const X = v => L + (v - x0) / (x1 - x0 || 1) * pw, Y = v => T + ph - (v - y0) / (y1 - y0 || 1) * ph;
  let s = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="${aria}"><line x1="${L}" x2="${W - Rm}" y1="${T + ph}" y2="${T + ph}" stroke="var(--axis)"/><line x1="${L}" x2="${L}" y1="${T}" y2="${T + ph}" stroke="var(--axis)"/>`;
  s += `<text class="tick" x="${L}" y="${H - 8}">${xtick(x0)}</text><text class="tick" x="${W - Rm}" y="${H - 8}" text-anchor="end">${xtick(x1)}</text><text class="lab" x="${(L + W) / 2}" y="${H - 8}" text-anchor="middle">${xl}</text>`;
  s += `<text class="lab" transform="translate(10 ${T + ph / 2}) rotate(-90)" text-anchor="middle">${yl}</text>`;
  s += `<text class="tick" x="${L - 4}" y="${T + 8}" text-anchor="end">${fmt(y1)}</text><text class="tick" x="${L - 4}" y="${T + ph}" text-anchor="end">${fmt(y0)}</text>`;
  pts.forEach((p, i) => s += `<circle data-i="${i}" cx="${X(p[0])}" cy="${Y(p[1])}" r="3.5" fill="${p[2] ? "var(--s2)" : "var(--s1)"}" fill-opacity=".75" stroke="var(--surface)" stroke-width="1"/>`);
  return s + `</svg>`; }
(function(){ const el = document.getElementById("scatter");
  el.innerHTML = scatter(D.scatter, 520, 300, "PC1", "PC2", "PC1 versus PC2 scores");
  el.querySelectorAll("circle").forEach(c => { c.addEventListener("mousemove", e => { const p = D.scatter[+c.dataset.i];
    showTip(e, `${p[3]}<br>PC1 ${p[0].toFixed(2)} · PC2 ${p[1].toFixed(2)}<br>${p[2] ? "small tissue" : "adequate tissue"}`); }); c.addEventListener("mouseleave", hide); }); })();
// 2c. loadings heatmap
(function(){ const f = D.eng_features, k = D.keep, cell = 46, rowh = 26, left = Math.max(...f.map(x => x.length)) * 6.3 + 16, top = 22;
  const css = getComputedStyle(document.documentElement), hex = v => { const c = css.getPropertyValue(v).trim(); return [1, 3, 5].map(i => parseInt(c.slice(i, i + 2), 16)); };
  const mid = hex("--mid"), pos = hex("--pos"), neg = hex("--neg");
  const mix = r => { const t = Math.min(1, Math.abs(r) / 0.7), e = r >= 0 ? pos : neg; return `rgb(${mid.map((m, j) => Math.round(m + (e[j] - m) * t)).join(",")})`; };
  let s = `<svg viewBox="0 0 ${left + k * cell + 8} ${top + f.length * rowh + 4}" style="min-width:${left + k * 36}px;max-width:${left + k * cell + 8}px" role="img" aria-label="PCA loadings">`;
  for (let c = 0; c < k; c++) s += `<text class="lab" x="${left + c * cell + cell / 2}" y="14" text-anchor="middle">PC${c + 1}</text>`;
  f.forEach((name, i) => { s += `<text class="lab" x="${left - 8}" y="${top + i * rowh + rowh / 2 + 4}" text-anchor="end">${name}</text>`;
    for (let c = 0; c < k; c++) { const v = D.loadings[i][c];
      s += `<rect x="${left + c * cell + 1}" y="${top + i * rowh + 1}" width="${cell - 2}" height="${rowh - 2}" rx="3" fill="${mix(v)}"><title>${name} on PC${c + 1}: ${v.toFixed(2)}</title></rect>`;
      if (Math.abs(v) >= 0.3) s += `<text x="${left + c * cell + cell / 2}" y="${top + i * rowh + rowh / 2 + 4}" text-anchor="middle" style="font-size:10px;fill:#ffffff;pointer-events:none">${v.toFixed(2)}</text>`; } });
  document.getElementById("load").innerHTML = s + `</svg>`; })();
// 4. tissue dependence small multiples
document.getElementById("tissub").textContent = `Each engineered feature (z-score) against analysed area (${R.area_column}, log scale; axis ends show the smallest and largest area). Small tissue: ${R.n_small} of ${D.n} rows (${R.n_small_area} below ${R.min_area} mm², ${R.n_small_regions} with fewer than ${R.min_regions} tumor regions). ρ = Spearman correlation, all rows / adequate-tissue rows only.`;
(function(){ const box = document.getElementById("tissue");
  D.eng_features.forEach((f, j) => { const d = D.dependence[j];
    const pts = D.tissue.filter(t => t[1][j] !== null).map(t => [t[0], t[1][j], t[2]]);
    const card = document.createElement("div"); card.className = "card";
    card.innerHTML = `<h3>${f}</h3><div class="meta">ρ ${d.rho_all === null ? "–" : d.rho_all.toFixed(2)} all · ${d.rho_adequate === null ? "–" : d.rho_adequate.toFixed(2)} adequate</div>` +
      scatter(pts, 260, 170, "", "z", `${f} versus log analysed area`, x => fmt(Math.exp(x)) + " mm²");
    box.appendChild(card); }); })();
</script></body></html>"""


if __name__ == "__main__":
    main()
