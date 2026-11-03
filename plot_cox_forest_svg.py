#!/usr/bin/env python3
r"""Forest plots (SVG) of the hazard ratios from cox_features.py.

    python plot_cox_forest_svg.py cox_univariable.csv cox_adjusted.csv cox_by_stage.csv \
        cox_strata.csv cox_mv.csv --output-dir plots

One SVG per results file and model: <file>__<model>.svg. Cohorts from the
same file (e.g. stage I-II and III-IV from --by) are drawn side by side and
share the feature rows and the HR axis, so they can be compared directly.

Each row: a feature (readable name), its HR with 95% CI on a log axis
(dashed line at 1 = no effect), and the text "HR (95% CI)  p". Filled
square: p < 0.05 (--alpha); hollow: not significant. --significance q fills
on the BH q-value instead, and --show-q adds the q column. A CI running past the axis ends
in an arrow. Rows are grouped by feature set (tumor morphology, TIL / TSR,
immune proximity, necrosis, other). Multivariable models also show their
covariates (stage, age, ...) in their own group. Rows that could not be
estimated say why.

--sort input|hr|p orders rows within each group (default input order).
Feature names are shown as readable labels (e.g. log_necrosis_tissue_fraction
-> "Necrosis fraction (log)"); --labels FILE.csv (columns feature,label)
overrides or adds names, --raw-names shows the column names.
SVG files open in any browser and import into PowerPoint / Illustrator /
Inkscape. Standard library only.
"""
import argparse
import csv
import math
import re
import sys
from collections import OrderedDict, defaultdict
from pathlib import Path
from xml.sax.saxutils import escape

INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SIG = "#2a78d6"
FONT = "system-ui, -apple-system, 'Segoe UI', Helvetica, Arial, sans-serif"
TICKS = [0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50]
# Checked in this order; the first match wins (TIL names can contain "tumor").
SECTIONS = [
    ("TIL / tumor-stroma ratio", re.compile(r"tsr|til_stromal|til_intratumoral|til_between")),
    ("Immune proximity", re.compile(r"til_|immune_")),
    ("Necrosis", re.compile(r"necrosis")),
    ("Tumor morphology", re.compile(r"(^|_)tumor_|elongation|^tumor")),
]
DISPLAY_ORDER = ["Tumor morphology", "TIL / tumor-stroma ratio", "Immune proximity", "Necrosis"]


LABELS = {
    "log_tumor_area_mm2": "Tumor area (mm², log)",
    "tumor_area_fraction": "Tumor area fraction",
    "log_tumor_edge_per_area_per_mm": "Tumor edge per area (log)",
    "log_tumor_islands_per_mm2": "Tumor islands per mm² (log)",
    "tumor_largest_island_share": "Largest tumor island share",
    "sqrt_tumor_island_roundness": "Tumor island roundness (√)",
    "tumor_island_solidity": "Tumor island solidity",
    "log_elongation_minus1": "Tumor island elongation (log)",
    "tumor_edge_irregularity": "Tumor edge irregularity",
    "log_tumor_island_spacing_um": "Tumor island spacing (µm, log)",
    "tumor_island_gap_um": "Tumor island gap (µm)",
    "fragmentation_score": "Tumor fragmentation score",
    "log_tsr_ratio": "Tumor-to-stroma ratio (log)",
    "tsr_stroma_high": "Stroma-high (TSR < 1)",
    "sqrt_til_stromal_pct": "Stromal TILs (%, √)",
    "log_til_intratumoral_pct": "Intratumoral TILs (%, log)",
    "log_til_between_tumor_pct": "TILs between tumor regions (%, log)",
    "log_til_total_area_mm2": "Total TIL area (mm², log)",
    "logit_til_share_inside_tumor": "TILs inside tumor (share, logit)",
    "logit_til_share_touching_tumor": "TILs touching tumor (share, logit)",
    "til_within_50um_pct": "TILs within 50 µm of tumor (%)",
    "sqrt_til_beyond_100um_pct": "TILs beyond 100 µm of tumor (%, √)",
    "sqrt_til_beyond_200um_pct": "TILs beyond 200 µm of tumor (%, √)",
    "til_outside_distance_median_um": "Median TIL distance to tumor (µm)",
    "immune_phenotype_peritumoral": "Peritumoral phenotype (vs margin-localized)",
    "immune_phenotype_immune_penetrated": "Immune-penetrated phenotype (vs margin-localized)",
    "log_necrosis_tissue_fraction": "Necrosis fraction (log)",
    "log_necrosis_area_mm2": "Necrosis area (mm², log)",
    "log_necrosis_boundary_length_mm": "Necrosis boundary length (mm, log)",
    "necrosis_present": "Necrosis present (fraction ≥ 0.05)",
    "AGE": "Age (per year)",
}


def readable(name, overrides, raw):
    """Display label: --labels override, built-in label, or a tidied column name."""
    if name in overrides:
        return overrides[name]
    if raw:
        return name
    if name in LABELS:
        return LABELS[name]
    m = re.match(r"^(\S+) (I{1,3}V?|IV)(-I{1,3}V?|-IV)? vs (I{1,3}V?|IV)(-I{1,3}V?)?$", name)
    if m and m.group(1).lower() == "stage":
        return "Stage " + name.split(" ", 1)[1].replace("-", "–")
    if name.lower().startswith("stage") and "ordinal" in name:
        return "Stage (per step, I–IV)"
    return name.replace("_", " ").replace(" pct", " (%)").replace(" um", " (µm)").strip().capitalize()


def section_of(feature):
    for name, pattern in SECTIONS:
        if pattern.search(feature):
            return name
    return "Other"


def num(v):
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def fmt_hr(x):
    return "–" if x is None else f"{x:.2f}" if x < 100 else f"{x:.0f}"


def fmt_p(p):
    if p is None:
        return "–"
    # 3 significant figures below 0.1, so 0.0499 is not shown as 0.050 next to a filled square
    return "<0.001" if p < 0.001 else f"{p:.3g}" if p < 0.1 else f"{p:.2f}"


def slug(text):
    return re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_").lower()


def read_results(path):
    with open(path, newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    need = {"cohort", "model", "feature", "term", "hr", "ci_low", "ci_high", "p", "q"}
    if not rows or not need <= set(rows[0]):
        sys.exit(f"{path}: not a cox_features.py results file (needs {sorted(need)})")
    return rows


def draw(model, cohorts, labels, alpha, sort, significance="p", show_q=False, raw_names=False):
    """cohorts: OrderedDict cohort -> list of result rows. Returns SVG text."""
    # Rows: features (and covariates for multivariable models), grouped by section.
    multivariable = model.startswith("multivariable")
    order = OrderedDict()
    for rows in cohorts.values():
        for r in rows:
            is_cov = r["feature"] == "(covariate)"
            if r["feature"] == "(all)":
                continue
            name = r["term"] if is_cov else r["feature"]
            if is_cov and not multivariable:
                continue
            key = ("Covariates" if is_cov else section_of(name), name)
            order.setdefault(key, None)
    sections = OrderedDict()
    for sec, name in order:
        sections.setdefault(sec, []).append(name)
    section_rank = DISPLAY_ORDER + ["Other", "Covariates"]
    sections = OrderedDict(sorted(sections.items(), key=lambda kv: section_rank.index(kv[0])))

    lookup = {}
    for cohort, rows in cohorts.items():
        for r in rows:
            name = r["term"] if r["feature"] == "(covariate)" else r["feature"]
            if r["feature"] == "(all)":
                lookup[(cohort, "(all)")] = r
            elif r["term"] == name or r["feature"] == "(covariate)":
                lookup[(cohort, name)] = r
    first = next(iter(cohorts))
    if sort != "input":
        for sec, names in sections.items():
            def k(n):
                r = lookup.get((first, n), {})
                v = num(r.get("hr")) if sort == "hr" else num(r.get("p"))
                return (v is None, -(abs(math.log(v))) if (sort == "hr" and v) else (v if v is not None else 0))
            names.sort(key=k)

    # Shared log axis across cohorts.
    values = [v for r in lookup.values() for v in (num(r.get("ci_low")), num(r.get("ci_high")), num(r.get("hr"))) if v]
    lo = min(values + [1.0]) / 1.2 if values else 0.5
    hi = max(values + [1.0]) * 1.2 if values else 2.0
    lo, hi = max(lo, 0.05), min(hi, 50)
    ticks = [t for t in TICKS if lo <= t <= hi]
    if len(ticks) < 3:
        ticks = sorted(set(ticks + [t for t in (0.5, 1, 2) if lo <= t <= hi] or [1]))

    names_shown = [readable(n, labels, raw_names) for names in sections.values() for n in names]
    row_h, plot_w, gap = 24, 250, 28
    label_w = max(260, int(max([len(n) for n in names_shown] + [20]) * 6.4) + 44)
    text_w = 190 if show_q else 150
    top = 98
    n_rows = sum(len(v) + 1 for v in sections.values())
    panel_w = plot_w + text_w
    width = label_w + len(cohorts) * (panel_w + gap) + 8
    height = top + n_rows * row_h + 56
    X = lambda panel, v: label_w + panel * (panel_w + gap) + (math.log(v) - math.log(lo)) / (math.log(hi) - math.log(lo)) * plot_w

    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" '
           f'font-family="{FONT}" role="img" aria-label="Forest plot: {escape(model)}">',
           f'<rect width="{width}" height="{height}" fill="#ffffff"/>',
           f'<text x="16" y="26" font-size="16" font-weight="600" fill="{INK}">Cox hazard ratios: {escape(model)}</text>',
           f'<text x="16" y="46" font-size="12" fill="{INK2}">HR per 1 SD (0/1 features: 1 vs 0); HR &gt; 1 = worse survival. '
           f'Filled = {"BH q" if significance == "q" else "p"} &lt; {alpha:g}.</text>']

    for panel, (cohort, rows) in enumerate(cohorts.items()):
        x0 = label_w + panel * (panel_w + gap)
        any_row = next((r for r in rows if num(r.get("n"))), {})
        out.append(f'<text x="{x0}" y="70" font-size="13" fill="{INK2}"><tspan font-weight="600" fill="{INK}">'
                   f'{escape(cohort)}</tspan>   n = {escape(any_row.get("n", "–"))}, '
                   f'events = {escape(any_row.get("events", "–"))}</text>')
        y_axis = top + n_rows * row_h
        for t in ticks:
            dash = ' stroke-dasharray="4 3"' if t == 1 else ""
            out.append(f'<line x1="{X(panel, t):.1f}" x2="{X(panel, t):.1f}" y1="{top - 6}" y2="{y_axis}" '
                       f'stroke="{GRID if t != 1 else MUTED}" stroke-width="1"{dash}/>')
            out.append(f'<text x="{X(panel, t):.1f}" y="{y_axis + 16}" font-size="10" fill="{MUTED}" '
                       f'text-anchor="middle">{t:g}</text>')
        out.append(f'<line x1="{x0}" x2="{x0 + plot_w}" y1="{y_axis}" y2="{y_axis}" stroke="{AXIS}"/>')
        out.append(f'<text x="{x0 + plot_w / 2}" y="{y_axis + 34}" font-size="11" fill="{INK2}" text-anchor="middle">'
                   f'Hazard ratio (log scale)</text>')
        out.append(f'<text x="{x0 + plot_w + 8}" y="{top - 8}" font-size="10" fill="{MUTED}">HR (95% CI)</text>')
        out.append(f'<text x="{x0 + plot_w + 116}" y="{top - 8}" font-size="10" fill="{MUTED}">p</text>')
        if show_q:
            out.append(f'<text x="{x0 + plot_w + 156}" y="{top - 8}" font-size="10" fill="{MUTED}">q</text>')

    y = top
    for sec, names in sections.items():
        out.append(f'<text x="16" y="{y + 16}" font-size="12" font-weight="600" fill="{INK}">{escape(sec)}</text>')
        y += row_h
        for name in names:
            label = readable(name, labels, raw_names)
            out.append(f'<text x="28" y="{y + 16}" font-size="12" fill="{INK2}">{escape(label)}</text>')
            cy = y + row_h / 2
            for panel, cohort in enumerate(cohorts):
                x0 = label_w + panel * (panel_w + gap)
                r = lookup.get((cohort, name))
                hr, cl, ch = (num(r.get(k)) for k in ("hr", "ci_low", "ci_high")) if r else (None, None, None)
                if hr is None or cl is None or ch is None:
                    note = (r or {}).get("note") or (lookup.get((cohort, "(all)"), {}).get("note")) or "not estimated"
                    out.append(f'<text x="{x0 + 4}" y="{y + 16}" font-size="11" fill="{MUTED}">{escape(note[:48])}</text>')
                    continue
                q, p = num(r.get("q")), num(r.get("p"))
                value = q if significance == "q" else p
                sig = value is not None and value < alpha
                a, b = max(cl, lo), min(ch, hi)
                out.append(f'<line x1="{X(panel, a):.1f}" x2="{X(panel, b):.1f}" y1="{cy}" y2="{cy}" stroke="{SIG if sig else INK2}" stroke-width="2"/>')
                for edge, clipped, sign in ((a, cl < lo, -1), (b, ch > hi, 1)):
                    xe = X(panel, edge)
                    if clipped:
                        out.append(f'<path d="M{xe - sign * 6:.1f},{cy - 4} L{xe:.1f},{cy} L{xe - sign * 6:.1f},{cy + 4}" '
                                   f'fill="none" stroke="{SIG if sig else INK2}" stroke-width="2"/>')
                xm = X(panel, min(max(hr, lo), hi))
                fill = SIG if sig else "#ffffff"
                out.append(f'<rect x="{xm - 5:.1f}" y="{cy - 5}" width="10" height="10" rx="2" fill="{fill}" '
                           f'stroke="{SIG if sig else INK2}" stroke-width="2"><title>{escape(label)} · {escape(cohort)}: '
                           f'HR {fmt_hr(hr)} ({fmt_hr(cl)}–{fmt_hr(ch)}), p {fmt_p(p)}, q {fmt_p(q)}</title></rect>')
                weight = ' font-weight="600"' if sig else ""
                out.append(f'<text x="{x0 + plot_w + 8}" y="{y + 16}" font-size="11" fill="{INK}"{weight} '
                           f'font-variant-numeric="tabular-nums">{fmt_hr(hr)} ({fmt_hr(cl)}–{fmt_hr(ch)})</text>')
                out.append(f'<text x="{x0 + plot_w + 116}" y="{y + 16}" font-size="11" fill="{INK2}">{fmt_p(p)}</text>')
                if show_q:
                    out.append(f'<text x="{x0 + plot_w + 156}" y="{y + 16}" font-size="11" fill="{INK2}">{fmt_p(q)}</text>')
            y += row_h
    notes = sorted({r.get("note", "") for rows in cohorts.values() for r in rows if r.get("note")})
    if notes:
        out.append(f'<text x="16" y="{height - 8}" font-size="11" fill="{MUTED}">Note: {escape("; ".join(notes)[:160])}</text>')
    out.append("</svg>")
    return "\n".join(out)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("results", nargs="+", type=Path, help="cox_features.py output CSV(s)")
    parser.add_argument("--output-dir", type=Path, default=Path("plots"))
    parser.add_argument("--alpha", type=float, default=0.05, help="Significance threshold (default 0.05)")
    parser.add_argument("--significance", choices=["p", "q"], default="p",
                        help="Fill squares when p (default) or BH q is below --alpha")
    parser.add_argument("--show-q", action="store_true", help="Add the BH q column")
    parser.add_argument("--raw-names", action="store_true", help="Show column names instead of labels")
    parser.add_argument("--sort", choices=["input", "hr", "p"], default="input")
    parser.add_argument("--labels", type=Path, help="CSV with columns feature,label")
    args = parser.parse_args()

    labels = {}
    if args.labels:
        with open(args.labels, newline="", encoding="utf-8-sig") as handle:
            labels = {r["feature"]: r["label"] for r in csv.DictReader(handle)}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for path in args.results:
        rows = read_results(path)
        by_model = OrderedDict()
        for r in rows:
            by_model.setdefault(r["model"], OrderedDict()).setdefault(r["cohort"], []).append(r)
        for model, cohorts in by_model.items():
            svg = draw(model, cohorts, labels, args.alpha, args.sort, args.significance,
                       args.show_q, args.raw_names)
            out = args.output_dir / f"{path.stem}__{slug(model)}.svg"
            out.write_text(svg, encoding="utf-8")
            written.append((out, len(cohorts)))
    for out, n in written:
        print(f"Wrote {out}" + (f" ({n} cohorts side by side)" if n > 1 else ""))


if __name__ == "__main__":
    main()
