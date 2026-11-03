#!/usr/bin/env python3
r"""Interactive distribution page for each feature CSV (one HTML per input).

For every numeric feature: a histogram per group (each scaled to its group's
size), group medians, skewness before/after log1p, extreme outliers (> 3 IQR),
missing values and, with two groups, a Mann-Whitney p. Text features (e.g.
WSI_TSR_category, wsi_dominant_immune_phenotype) get category shares per
group. A Spearman correlation map shows redundant features; a table view
lists every statistic. A Raw / log1p toggle redraws the histograms.

Usage:
    python plot_feature_distributions.py all_groups_tils_tsr.csv \
        all_groups_immune_proximity.csv all_groups_necrosis_feature.csv --output-dir plots
    -> plots/all_groups_tils_tsr_distributions.html, ...

Input: one row per subject (or per WSI) with a group column, e.g. the output of
concat_group_outputs.py or aggregate_subjects.py. ID columns (group,
subject_id, sample_id, wsi_name, n_wsi) and clinical/survival columns (AGE,
OS_MONTHS, OS_STATUS, stage, sex, ...) are not plotted; --features picks
columns explicitly (wildcards allowed). Up to three groups. Standard library
only; the page opens in any browser, offline.
"""
import argparse
import csv
import fnmatch
import json
import math
import statistics as st
import sys
from collections import Counter, defaultdict
from pathlib import Path

IDS = {"group", "subject_id", "sample_id", "wsi_name", "n_wsi"}
MISSING = {"", "na", "nan", "none", "null"}
PATIENT_PATTERNS = ["age", "diagnosis age", "os_*", "os status", "dfs_*", "pfs_*", "dss_*",
                    "*months*", "*status*", "*days*", "*years*", "*stage*", "*event*", "sex",
                    "gender", "vital_status", "*survival*", "tcga_case_id", "patient_id",
                    "case_id", "source_file"]


def is_missing(v):
    return v is None or v.strip().lower() in MISSING


def num(v):
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def matches(column, patterns):
    return any(fnmatch.fnmatchcase(column.lower(), p.lower()) for p in patterns)


def skew(x):
    if len(x) < 3:
        return 0.0
    m, s = st.fmean(x), st.pstdev(x)
    return 0.0 if s == 0 else sum((v - m) ** 3 for v in x) / len(x) / s ** 3


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


def pearson(a, b):
    if len(a) < 3:
        return 0.0
    ma, mb = st.fmean(a), st.fmean(b)
    sa = math.sqrt(sum((v - ma) ** 2 for v in a))
    sb = math.sqrt(sum((v - mb) ** 2 for v in b))
    return 0.0 if sa == 0 or sb == 0 else sum((x - ma) * (y - mb) for x, y in zip(a, b)) / sa / sb


def mwu_p(a, b):
    """Two-sided Mann-Whitney U p-value (normal approximation, tie-corrected)."""
    if not a or not b:
        return None
    allv = a + b
    r, n1, n2 = ranks(allv), len(a), len(b)
    u = sum(r[:n1]) - n1 * (n1 + 1) / 2
    n = n1 + n2
    ties = Counter(allv)
    var = n1 * n2 / 12 * ((n + 1) - sum(t ** 3 - t for t in ties.values()) / (n * (n - 1)))
    if var <= 0:
        return 1.0
    z = max(0.0, abs(u - n1 * n2 / 2) - 0.5) / math.sqrt(var)
    return math.erfc(z / math.sqrt(2))


def bh(pvalues):
    """Benjamini-Hochberg q-values, None kept as None."""
    idx = sorted((p, i) for i, p in enumerate(pvalues) if p is not None)
    q, prev = [None] * len(pvalues), 1.0
    for rank, (p, i) in reversed(list(enumerate(idx, 1))):
        prev = min(prev, p * len(idx) / rank)
        q[i] = prev
    return q


def analyse(path, feature_patterns=None):
    with open(path, newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        header = [c.strip() for c in reader.fieldnames or []]
        rows = [{k.strip(): (v or "") for k, v in r.items() if k is not None} for r in reader]
    if not rows:
        sys.exit(f"{path}: no data rows")
    if "group" not in header:
        for r in rows:
            r["group"] = "all"
    groups = sorted({r["group"] for r in rows})
    if len(groups) > 3:
        sys.exit(f"{path}: {len(groups)} groups ({groups}); the page supports up to 3 - "
                 "filter the file or plot groups in separate runs")

    candidates = [c for c in header if c not in IDS]
    if feature_patterns:
        candidates = [c for c in candidates if matches(c, feature_patterns)]
        if not candidates:
            sys.exit(f"{path}: --features matched no columns")
    else:
        candidates = [c for c in candidates if not matches(c, PATIENT_PATTERNS)]

    numeric, text, skipped = [], [], []
    for c in candidates:
        filled = [r[c] for r in rows if not is_missing(r[c])]
        if not filled:
            skipped.append(f"{c} (empty)")
        elif all(num(v) is not None for v in filled):
            (numeric if len({num(v) for v in filled}) > 1 else skipped).append(
                c if len({num(v) for v in filled}) > 1 else f"{c} (constant)")
        elif len(set(filled)) > 1:
            text.append(c)
        else:
            skipped.append(f"{c} (constant)")

    stats, data = [], {}
    for f in numeric:
        pts = [(r["group"], num(r[f])) for r in rows]
        vals = [v for _, v in pts if v is not None]
        q1, med, q3 = st.quantiles(vals, n=4, method="inclusive") if len(vals) > 1 else (vals[0],) * 3
        iqr, lo = q3 - q1, min(vals)
        logv = [math.log1p(v - lo) if lo < 0 else math.log1p(v) for v in vals]
        by = {g: [v for gg, v in pts if gg == g and v is not None] for g in groups}
        stats.append(dict(feature=f, n=len(vals), missing=len(rows) - len(vals),
                          zeros=sum(v == 0 for v in vals), min=lo, median=med, max=max(vals),
                          skew=skew(vals), log_skew=skew(logv),
                          extreme_outliers=sum(v < q1 - 3 * iqr or v > q3 + 3 * iqr for v in vals),
                          group_medians={g: (st.median(v) if v else None) for g, v in by.items()},
                          p=mwu_p(by[groups[0]], by[groups[1]]) if len(groups) == 2 else None))
        data[f] = [[g, v] for g, v in pts if v is not None]
    for s, q in zip(stats, bh([s["p"] for s in stats])):
        s["q"] = q

    categories = []
    for f in text:
        shares = {}
        for g in groups:
            values = [r[f] for r in rows if r["group"] == g and not is_missing(r[f])]
            shares[g] = {k: v / len(values) for k, v in Counter(values).items()} if values else {}
        levels = [k for k, _ in Counter(r[f] for r in rows if not is_missing(r[f])).most_common()]
        categories.append(dict(feature=f, levels=levels[:12], shares=shares,
                               missing=sum(is_missing(r[f]) for r in rows)))

    corr = []
    for a in numeric:
        line = []
        for b in numeric:
            pairs = [(num(r[a]), num(r[b])) for r in rows]
            pairs = [p for p in pairs if p[0] is not None and p[1] is not None]
            line.append(pearson(ranks([p[0] for p in pairs]), ranks([p[1] for p in pairs])))
        corr.append(line)

    per_subject = len({(r["group"], r.get("subject_id", i)) for i, r in enumerate(rows)}) == len(rows)
    return dict(title=Path(path).stem, groups=groups,
                counts={g: sum(r["group"] == g for r in rows) for g in groups},
                unit="subject" if per_subject else "WSI", features=numeric, stats=stats,
                data=data, corr=corr, categories=categories, skipped=skipped)


TEMPLATE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__ Distributions</title>
<style>
:root{color-scheme:light;--page:#f9f9f7;--surface:#fcfcfb;--ink:#0b0b0b;--ink-2:#52514e;--muted:#898781;
  --grid:#e1e0d9;--axis:#c3c2b7;--ring:rgba(11,11,11,.10);--s1:#2a78d6;--s2:#eb6834;--s3:#1baf7a;
  --pos:#e34948;--neg:#2a78d6;--mid:#f0efec}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){color-scheme:dark;--page:#0d0d0d;--surface:#1a1a19;
  --ink:#f0efec;--ink-2:#c3c2b7;--muted:#898781;--grid:#2c2c2a;--axis:#383835;--ring:rgba(255,255,255,.10);
  --s1:#3987e5;--s2:#d95926;--s3:#199e70;--pos:#e66767;--neg:#3987e5;--mid:#383835}}
:root[data-theme="dark"]{color-scheme:dark;--page:#0d0d0d;--surface:#1a1a19;--ink:#f0efec;--ink-2:#c3c2b7;
  --muted:#898781;--grid:#2c2c2a;--axis:#383835;--ring:rgba(255,255,255,.10);--s1:#3987e5;--s2:#d95926;
  --s3:#199e70;--pos:#e66767;--neg:#3987e5;--mid:#383835}
*{box-sizing:border-box}
body{margin:0;background:var(--page);color:var(--ink);font:14px/1.45 system-ui,-apple-system,"Segoe UI",sans-serif}
main{max-width:1180px;margin:0 auto;padding:24px 16px 48px}
h1{font-size:22px;margin:0 0 4px;overflow-wrap:anywhere}h2{font-size:17px;margin:32px 0 4px}
.sub{color:var(--ink-2);margin:0 0 16px}
.bar{display:flex;flex-wrap:wrap;gap:16px;align-items:center;margin:0 0 16px}
.legend{display:flex;gap:16px;flex-wrap:wrap;color:var(--ink-2)}
.legend span{display:inline-flex;align-items:center;gap:6px}
.sw{width:16px;height:3px;border-radius:2px;display:inline-block;vertical-align:middle}
.seg{display:inline-flex;border:1px solid var(--ring);border-radius:8px;overflow:hidden}
.seg button{font:inherit;color:var(--ink-2);background:var(--surface);border:0;padding:6px 12px;cursor:pointer}
.seg button[aria-pressed="true"]{color:var(--ink);font-weight:600;background:var(--mid)}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(340px,1fr));gap:12px}
@media (max-width:420px){.grid{grid-template-columns:1fr}}
.card{background:var(--surface);border:1px solid var(--ring);border-radius:10px;padding:12px 12px 8px;min-width:0}
.card h3{font-size:14px;margin:0;overflow-wrap:anywhere}
.meta{color:var(--ink-2);font-size:12px;margin:2px 0 4px;font-variant-numeric:tabular-nums}
.chip{display:inline-block;font-size:11px;padding:1px 7px;border-radius:999px;border:1px solid var(--ring);color:var(--ink-2);margin-left:6px;font-weight:400}
svg{display:block;width:100%;height:auto}
.tick{fill:var(--muted);font-size:10px;font-variant-numeric:tabular-nums}
.lab{fill:var(--ink-2);font-size:11px}
.tip{position:fixed;pointer-events:none;background:var(--surface);color:var(--ink);border:1px solid var(--ring);border-radius:8px;
  padding:6px 9px;font-size:12px;box-shadow:0 4px 14px rgba(0,0,0,.12);display:none;z-index:5;font-variant-numeric:tabular-nums;white-space:nowrap}
.heat{background:var(--surface);border:1px solid var(--ring);border-radius:10px;padding:12px;overflow-x:auto}
table{border-collapse:collapse;width:100%;font-size:12px;font-variant-numeric:tabular-nums}
th,td{padding:5px 8px;border-bottom:1px solid var(--grid);text-align:right;white-space:nowrap}
th:first-child,td:first-child{text-align:left}
details{margin-top:24px;background:var(--surface);border:1px solid var(--ring);border-radius:10px;padding:10px 12px}
.tablewrap{overflow-x:auto} summary{cursor:pointer;font-weight:600}
</style></head><body><main>
<h1 id="title"></h1><p class="sub" id="subtitle"></p>
<div class="bar"><div class="legend" id="legend"></div>
<div class="seg" role="group" aria-label="Scale"><button type="button" data-scale="raw" aria-pressed="true">Raw</button><button type="button" data-scale="log" aria-pressed="false">log1p</button></div></div>
<div class="grid" id="grid"></div>
<div id="catsec"><h2>Text features (share of each group)</h2><div class="grid" id="cats"></div></div>
<div id="heatsec"><h2>Feature redundancy (Spearman ρ)</h2>
<p class="sub">Columns follow the row order. Red = rank together, blue = opposite; values shown where |ρ| ≥ 0.7, hover any cell for its value.</p>
<div class="heat" id="heat"></div></div>
<details><summary>Table view</summary><div class="tablewrap" id="table"></div></details>
<p class="sub" id="skipped" style="margin-top:16px"></p>
</main><div class="tip" id="tip"></div>
<script>
const D = __DATA__;
const colors = ["var(--s1)","var(--s2)","var(--s3)"];
const tip = document.getElementById("tip");
const fmt = v => { if (v === null || v === undefined) return "–"; const a = Math.abs(v);
  return a === 0 ? "0" : a >= 1000 ? v.toFixed(0) : a >= 10 ? v.toFixed(1) : a >= 0.1 ? v.toFixed(3) : v.toPrecision(3); };
const pfmt = p => p === null || p === undefined ? "–" : p < 0.001 ? "<0.001" : p.toFixed(3);
const label = g => g.replace(/_/g, " ");
function showTip(e, html){ tip.innerHTML = html; tip.style.display = "block";
  tip.style.left = Math.min(e.clientX + 14, innerWidth - tip.offsetWidth - 8) + "px";
  tip.style.top = Math.min(e.clientY + 14, innerHeight - tip.offsetHeight - 8) + "px"; }
function hideTip(){ tip.style.display = "none"; }
document.getElementById("title").textContent = D.title.replace(/_/g, " ") + " – feature distributions";
document.getElementById("subtitle").textContent = D.groups.map(g => `${label(g)} n = ${D.counts[g]}`).join(" · ") +
  ` · one row per ${D.unit}` + (D.groups.length > 1 ? " · each group's histogram is scaled to its own size" : "");
document.getElementById("legend").innerHTML = D.groups.length > 1 ? D.groups.map((g, i) =>
  `<span><i class="sw" style="background:${colors[i]}"></i>${label(g)}</span>`).join("") : "";
if (D.skipped.length) document.getElementById("skipped").textContent = "Not plotted: " + D.skipped.join(", ");
const two = D.groups.length === 2;
function suggestion(s){
  if (Math.abs(s.skew) > 1 && Math.abs(s.log_skew) < Math.abs(s.skew) - 0.3) return "log candidate";
  if (Math.abs(s.skew) > 1) return "skewed";
  return "near-symmetric"; }
let scale = "raw";
function draw(){
  const grid = document.getElementById("grid"); grid.innerHTML = "";
  D.stats.forEach(s => {
    const f = s.feature, pts = D.data[f], lo = s.min;
    const tr = v => scale === "log" ? Math.log1p(lo < 0 ? v - lo : v) : v;
    const back = v => scale === "log" ? (lo < 0 ? Math.expm1(v) + lo : Math.expm1(v)) : v;
    const vals = pts.map(p => tr(p[1])), mn = Math.min(...vals), mx = Math.max(...vals), bins = 24, w = (mx - mn) / bins || 1;
    const hist = D.groups.map(g => { const c = new Array(bins).fill(0); let n = 0;
      pts.forEach(p => { if (p[0] !== g) return; n++; c[Math.min(bins - 1, Math.floor((tr(p[1]) - mn) / w))]++; });
      return {c, n}; });
    const W = 340, H = 150, L = 8, R = 8, T = 10, B = 26, pw = W - L - R, ph = H - T - B;
    const ymax = Math.max(...hist.flatMap(h => h.c.map(c => h.n ? c / h.n : 0)), 0.01);
    const X = v => L + (v - mn) / (mx - mn || 1) * pw, Y = v => T + ph - v / ymax * ph;
    let svg = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Distribution of ${f}"><line x1="${L}" x2="${W - R}" y1="${T + ph}" y2="${T + ph}" stroke="var(--axis)"/>`;
    hist.forEach((h, gi) => { if (!h.n) return; let d = `M${X(mn)},${Y(0)}`;
      h.c.forEach((c, i) => { const y = Y(c / h.n); d += `L${X(mn + i * w)},${y}L${X(mn + (i + 1) * w)},${y}`; });
      svg += `<path d="${d}L${X(mx)},${Y(0)}" fill="none" stroke="${colors[gi]}" stroke-width="2" stroke-linejoin="round"/>`;
      const med = s.group_medians[D.groups[gi]];
      if (med !== null) svg += `<line x1="${X(tr(med))}" x2="${X(tr(med))}" y1="${T + ph}" y2="${T + ph + 6}" stroke="${colors[gi]}" stroke-width="2"/>`; });
    [mn, (mn + mx) / 2, mx].forEach((v, i) =>
      svg += `<text class="tick" x="${X(v)}" y="${H - 6}" text-anchor="${["start","middle","end"][i]}">${fmt(back(v))}</text>`);
    for (let i = 0; i < bins; i++) svg += `<rect data-i="${i}" x="${X(mn + i * w)}" y="${T}" width="${pw / bins}" height="${ph + 8}" fill="transparent"/>`;
    const card = document.createElement("div"); card.className = "card";
    card.innerHTML = `<h3>${f}<span class="chip">${suggestion(s)}</span></h3><div class="meta">skew ${s.skew.toFixed(2)} → log ${s.log_skew.toFixed(2)} · extreme outliers ${s.extreme_outliers}` +
      (s.missing ? ` · missing ${s.missing}` : "") + (two ? ` · group p ${pfmt(s.p)} (q ${pfmt(s.q)})` : "") + `</div>` + svg + `</svg>`;
    card.querySelectorAll("rect").forEach(r => {
      r.addEventListener("mousemove", e => { const i = +r.dataset.i, a = mn + i * w;
        showTip(e, `<b>${fmt(back(a))} – ${fmt(back(a + w))}</b><br>` + D.groups.map((g, gi) =>
          `<i class="sw" style="background:${colors[gi]}"></i> ${label(g)}: ${hist[gi].c[i]} (${hist[gi].n ? (100 * hist[gi].c[i] / hist[gi].n).toFixed(0) : 0}%)`).join("<br>")); });
      r.addEventListener("mouseleave", hideTip); });
    grid.appendChild(card); }); }
document.querySelectorAll(".seg button").forEach(b => b.addEventListener("click", () => { scale = b.dataset.scale;
  document.querySelectorAll(".seg button").forEach(x => x.setAttribute("aria-pressed", x === b)); draw(); }));
draw();
// Text features: horizontal share bars, one row per category and group
(function(){ const box = document.getElementById("cats");
  if (!D.categories.length) { document.getElementById("catsec").style.display = "none"; return; }
  D.categories.forEach(c => { const rowH = 12, gap = 2, blockH = D.groups.length * (rowH + gap) + 10, W = 340, L = 130, R = 40;
    const H = c.levels.length * blockH + 8;
    let svg = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Category shares of ${c.feature}">`;
    c.levels.forEach((lv, li) => { const y0 = 4 + li * blockH;
      svg += `<text class="lab" x="${L - 8}" y="${y0 + (D.groups.length * (rowH + gap)) / 2 + 3}" text-anchor="end">${lv.length > 20 ? lv.slice(0, 19) + "…" : lv}</text>`;
      D.groups.forEach((g, gi) => { const v = c.shares[g][lv] || 0, y = y0 + gi * (rowH + gap), bw = Math.max(v * (W - L - R), v > 0 ? 2 : 0);
        svg += `<rect x="${L}" y="${y}" width="${bw}" height="${rowH}" rx="3" fill="${colors[gi]}"><title>${label(g)} · ${lv}: ${(100 * v).toFixed(0)}%</title></rect>`;
        svg += `<text class="tick" x="${L + bw + 4}" y="${y + rowH - 2}">${(100 * v).toFixed(0)}%</text>`; }); });
    const card = document.createElement("div"); card.className = "card";
    card.innerHTML = `<h3>${c.feature}</h3><div class="meta">${c.levels.length} categories` + (c.missing ? ` · missing ${c.missing}` : "") + `</div>` + svg + `</svg>`;
    box.appendChild(card); }); })();
// Correlation heatmap
(function(){ const f = D.features, n = f.length;
  if (n < 2) { document.getElementById("heatsec").style.display = "none"; return; }
  const cell = 30, left = Math.max(...f.map(x => x.length)) * 6.3 + 16, top = 8;
  const css = getComputedStyle(document.documentElement);
  const hex = v => { const c = css.getPropertyValue(v).trim(); return [1, 3, 5].map(i => parseInt(c.slice(i, i + 2), 16)); };
  const mid = hex("--mid"), pos = hex("--pos"), neg = hex("--neg");
  const mix = r => { const t = Math.min(1, Math.abs(r)), e = r >= 0 ? pos : neg; return `rgb(${mid.map((m, k) => Math.round(m + (e[k] - m) * t)).join(",")})`; };
  let svg = `<svg viewBox="0 0 ${left + n * cell + 8} ${top + n * cell + 8}" style="min-width:${left + n * 22}px" role="img" aria-label="Spearman correlation matrix">`;
  f.forEach((a, i) => { svg += `<text class="lab" x="${left - 8}" y="${top + i * cell + cell / 2 + 4}" text-anchor="end">${a}</text>`;
    f.forEach((b, j) => { const r = D.corr[i][j];
      svg += `<rect data-i="${i}" data-j="${j}" x="${left + j * cell + 1}" y="${top + i * cell + 1}" width="${cell - 2}" height="${cell - 2}" rx="3" fill="${mix(r)}"/>`;
      if (Math.abs(r) >= 0.7 && i !== j) svg += `<text x="${left + j * cell + cell / 2}" y="${top + i * cell + cell / 2 + 3}" text-anchor="middle" style="font-size:9px;fill:#ffffff;pointer-events:none">${r.toFixed(2).replace("0.", ".")}</text>`; }); });
  const heat = document.getElementById("heat"); heat.innerHTML = svg + `</svg>`;
  heat.querySelectorAll("rect").forEach(r => { r.addEventListener("mousemove", e => { const i = +r.dataset.i, j = +r.dataset.j;
    showTip(e, `${f[i]}<br>${f[j]}<br><b>ρ = ${D.corr[i][j].toFixed(2)}</b>`); }); r.addEventListener("mouseleave", hideTip); }); })();
document.getElementById("table").innerHTML = `<table><thead><tr><th>Feature</th><th>n</th><th>Missing</th><th>Min</th><th>Median</th><th>Max</th><th>Skew</th><th>Skew (log)</th><th>Extreme outliers</th>` +
  D.groups.map(g => `<th>Median ${label(g)}</th>`).join("") + (two ? `<th>Group p</th><th>BH q</th>` : "") + `</tr></thead><tbody>` +
  D.stats.map(s => `<tr><td>${s.feature}</td><td>${s.n}</td><td>${s.missing}</td><td>${fmt(s.min)}</td><td>${fmt(s.median)}</td><td>${fmt(s.max)}</td><td>${s.skew.toFixed(2)}</td><td>${s.log_skew.toFixed(2)}</td><td>${s.extreme_outliers}</td>` +
    D.groups.map(g => `<td>${fmt(s.group_medians[g])}</td>`).join("") + (two ? `<td>${pfmt(s.p)}</td><td>${pfmt(s.q)}</td>` : "") + `</tr>`).join("") + `</tbody></table>`;
</script></body></html>"""


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("inputs", nargs="+", type=Path, help="Feature CSV file(s)")
    parser.add_argument("--features", nargs="+", help="Columns to plot, names or wildcards")
    parser.add_argument("--output-dir", type=Path, help="Folder for the pages (default: next to each input)")
    args = parser.parse_args()
    for path in args.inputs:
        if not path.is_file():
            sys.exit(f"Input not found: {path}")
        result = analyse(path, args.features)
        folder = args.output_dir or path.parent
        folder.mkdir(parents=True, exist_ok=True)
        out = folder / f"{path.stem}_distributions.html"
        payload = json.dumps(result).replace("</", "<\\/")
        out.write_text(TEMPLATE.replace("__TITLE__", path.stem).replace("__DATA__", payload), encoding="utf-8")
        print(f"{path.name}: {len(result['features'])} numeric + {len(result['categories'])} text features, "
              f"groups {result['counts']} -> {out}")
        if result["skipped"]:
            print("  not plotted: " + ", ".join(result["skipped"]))
        for s in result["stats"]:
            flag = "log candidate" if abs(s["skew"]) > 1 and abs(s["log_skew"]) < abs(s["skew"]) - 0.3 else ""
            q = f" q={s['q']:.3g}" if s.get("q") is not None else ""
            print(f"  {s['feature']:45} skew {s['skew']:6.2f} -> log {s['log_skew']:6.2f}  "
                  f"outliers {s['extreme_outliers']:2}  missing {s['missing']:3}{q}  {flag}")
        for i, a in enumerate(result["features"]):
            for j in range(i + 1, len(result["features"])):
                if abs(result["corr"][i][j]) >= 0.8:
                    print(f"  redundant: {a} ~ {result['features'][j]} (rho {result['corr'][i][j]:.2f})")


if __name__ == "__main__":
    main()
