#!/usr/bin/env python3
r"""Combined Kaplan-Meier figures for the features with Cox p <= 0.05.

    python km_cox_significant.py --data features_median.csv \
        --cox cox_univariable.csv --cox-by-stage cox_by_stage.csv \
        --stage-groups I,II III,IV --limit 120 --output-dir plots

Writes SVGs and matching PNGs to --output-dir:

  km_significant_all_patients.svg
      one KM panel per feature whose Cox p <= --p in the overall cohort
      (cohort "All" of --cox), three panels per row
  km_significant_by_stage.svg
      one row per feature whose Cox p <= --p in any stage group of
      --cox-by-stage; columns = the stage groups (plus All patients unless
      --no-all), so every feature is shown in every stage

Each panel: KM curves (Low / High at the median of the feature, or tertiles
with --split tertile; 0/1 features as 1 vs 0), log-rank p, and the Cox HR and
p of that feature in that cohort. Higher risk is always orange (#E69F00),
lower risk is always blue (#0072B2), and the middle tertile is gray.
Risk direction comes from each panel's Cox HR: HR > 1 means higher values
have higher risk; HR < 1 means lower values have higher risk. If the panel
HR is unavailable, the overall HR is used and the fallback is noted.
HR = 1 or no usable HR leaves risk unassigned (neutral gray curves).
"Cox p ≤ 0.05" panels are marked with a dot in their title.
Number-at-risk table and medians as in km_plot_svg.py.

--data is the per-subject table the Cox models were fitted on (e.g. the
aggregated engineered features). Splitting a log/sqrt/z-scored feature at its
median gives the same groups as splitting the raw values, so the curves are
those of the original measurement. --model picks the Cox model to read when a
file holds several (default: the first one in the file).

--all plots every feature in the Cox files instead of only those with
p <= --p (files are then named km_all_features_*.svg). --per-feature also
writes one file per feature (km_feature_<name>.svg: All patients + each stage
group). A PNG is saved next to every SVG by default (see svg_to_png.py;
--png-scale, default 2). Use --no-png for SVG only; --png remains accepted.
Needs km_plot_svg.py, plot_cox_forest_svg.py and svg_to_png.py in the same
folder, plus the rendering dependencies required by svg_to_png.py.
"""
import argparse
import csv
import math
import sys
from pathlib import Path
from xml.sax.saxutils import escape

sys.path.insert(0, str(Path(__file__).resolve().parent))
from km_plot_svg import (FONT, INK, INK2, ROMAN, is_missing, nice_ticks, number,  # noqa: E402
                         panel_svg, parse_event, parse_stage, quantile)
from plot_cox_forest_svg import LABELS, readable  # noqa: E402

PANEL_W, GAP = 400, 0
PNG = {"scale": 0, "failures": []}   # configured in main()
LOW_RISK_COLOR = "#0072B2"
HIGH_RISK_COLOR = "#E69F00"
MIDDLE_COLOR = "#999999"
UNKNOWN_COLORS = ("#555555", "#999999", "#CCCCCC")
RISK_LEGEND = (
    "Orange = higher risk; blue = lower risk; middle tertile = gray.",
    "Risk: panel Cox HR, with overall HR fallback if unavailable; unassigned risk = gray.",
)


def risk_direction(cox_row, overall_row=None):
    """Return risk direction and provenance without inventing a missing HR."""
    for row, source in ((cox_row, "panel"), (overall_row, "overall")):
        hr = number((row or {}).get("hr"))
        if hr is not None and math.isfinite(hr) and hr > 0:
            return (1 if hr > 1 else -1 if hr < 1 else 0), source
    return 0, "unavailable"


def risk_style(labels, direction):
    """Keep the same risk-to-color mapping for median, binary and tertile splits."""
    if direction == 0:
        colors = ([UNKNOWN_COLORS[0], UNKNOWN_COLORS[-1]]
                  if len(labels) == 2 else list(UNKNOWN_COLORS))
        return [f"{label} · risk unassigned" for label in labels], colors
    risks = (["lower risk", "higher risk"] if len(labels) == 2
             else ["lower risk", "middle third", "higher risk"])
    colors = ([LOW_RISK_COLOR, HIGH_RISK_COLOR] if len(labels) == 2
              else [LOW_RISK_COLOR, MIDDLE_COLOR, HIGH_RISK_COLOR])
    if direction < 0:
        risks.reverse()
        colors.reverse()
    return [label if risk == "middle third" else f"{label} · {risk}"
            for label, risk in zip(labels, risks)], colors


def read_csv(path):
    with open(path, newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def cox_lookup(rows, model):
    """{cohort: {feature: row}} for the chosen model (feature terms only)."""
    models = list(dict.fromkeys(r["model"] for r in rows))
    model = model or (models[0] if models else None)
    if model not in models:
        sys.exit(f"Model {model!r} not in the Cox file; available: {models}")
    out = {}
    for r in rows:
        if r["model"] == model and r["feature"] == r["term"]:
            out.setdefault(r["cohort"], {})[r["feature"]] = r
    return model, out


def cox_note(r):
    if not r or number(r.get("hr")) is None:
        return "Cox: not estimated"
    p = number(r["p"])
    ptxt = "not estimated" if p is None else "<0.001" if p < 0.001 else f"{p:.3g}" if p < 0.1 else f"{p:.2f}"
    return f"Cox HR {number(r['hr']):.2f}, p {ptxt}"


def groups_for(rows, feature, split, time_col, event_col, limit):
    values = [number(r[feature]) for r in rows]
    present = [v for v in values if v is not None]
    is_binary = set(present) <= {0.0, 1.0}
    if is_binary:
        labels = ["0", "1"]
        assign = lambda v: None if v is None else ("1" if v == 1 else "0")
        how = "1 vs 0"
    elif split == "tertile":
        a, b = quantile(present, 1 / 3), quantile(present, 2 / 3)
        labels = ["Low third", "Middle third", "High third"]
        assign = lambda v: None if v is None else labels[0] if v < a else labels[1] if v < b else labels[2]
        how = "tertiles"
    else:
        cut = quantile(present, 0.5)
        labels = ["Low (< median)", "High (≥ median)"]
        assign = lambda v: None if v is None else labels[1] if v >= cut else labels[0]
        how = "median split"
    return labels, assign, how


def build_groups(rows, feature, assign, time_col, event_col, limit):
    groups = {}
    for r in rows:
        t, e, v = number(r[time_col]), parse_event(r[event_col]), number(r[feature])
        if t is None or e is None or t < 0:
            continue
        if limit and t > limit:
            t, e = limit, 0
        g = assign(v)
        if g is not None:
            groups.setdefault(g, ([], []))
            groups[g][0].append(t)
            groups[g][1].append(e)
    return groups


def write_svg(path, width, height, title, subtitle, parts):
    head = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" '
            f'font-family="{FONT}" role="img" aria-label="{escape(title)}">',
            f'<rect width="{width}" height="{height}" fill="#ffffff"/>',
            f'<text x="16" y="26" font-size="16" font-weight="600" fill="{INK}">{escape(title)}</text>']
    for i, line in enumerate(subtitle):
        head.append(f'<text x="16" y="{46 + 16 * i}" font-size="12" fill="{INK2}">{escape(line)}</text>')
    path.write_text("\n".join(head + parts + ["</svg>"]), encoding="utf-8")
    if PNG["scale"]:
        try:
            from svg_to_png import svg_to_png
            png, used = svg_to_png(path, scale=PNG["scale"])
            print(f"  PNG: {png} ({used})")
        except Exception as error:  # keep the SVG; report why the PNG is missing
            PNG["failures"].append(path.name)
            print(f"  PNG not written for {path.name}: {error}", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", required=True, type=Path, help="Per-subject features + survival CSV")
    parser.add_argument("--cox", type=Path, help="cox_features.py results for the whole cohort")
    parser.add_argument("--cox-by-stage", type=Path, help="cox_features.py results run with --by stage")
    parser.add_argument("--model", help="Cox model to read (default: first in each file)")
    parser.add_argument("--p", type=float, default=0.05, help="Select features with Cox p <= this (default 0.05)")
    parser.add_argument("--split", choices=["median", "tertile"], default="median")
    parser.add_argument("--color-by", choices=["risk"], default="risk",
                        help="Risk colors are fixed: orange = higher risk, blue = lower risk")
    parser.add_argument("--time", default="OS_MONTHS")
    parser.add_argument("--event", default="OS_STATUS")
    parser.add_argument("--limit", type=float, default=0, help="Censor at this many months (0 = none)")
    parser.add_argument("--stage-column", default="stage")
    parser.add_argument("--stage-groups", nargs="+", default=["I,II", "III,IV"], metavar="I,II")
    parser.add_argument("--no-all", action="store_true", help="Stage figure without the All patients column")
    parser.add_argument("--all", action="store_true", help="Plot every feature, not only those with p <= --p")
    parser.add_argument("--per-feature", action="store_true", help="Also write one file per feature")
    png_options = parser.add_mutually_exclusive_group()
    png_options.add_argument("--png", dest="png", action="store_true", help="Save each plot as PNG (default)")
    png_options.add_argument("--no-png", dest="png", action="store_false", help="Save SVG files only")
    parser.set_defaults(png=True)
    parser.add_argument("--png-scale", type=float, default=2, help="PNG pixel density (default 2)")
    parser.add_argument("--output-dir", type=Path, default=Path("plots"))
    args = parser.parse_args()
    if not math.isfinite(args.png_scale) or args.png_scale <= 0:
        parser.error("--png-scale must be a finite positive number")
    PNG["scale"] = args.png_scale if args.png else 0
    PNG["failures"] = []
    if args.all:
        args.p = float("inf")
    prefix = "km_all_features" if args.all else "km_significant"
    what = "all features" if args.all else None
    if not args.cox and not args.cox_by_stage:
        parser.error("give --cox and/or --cox-by-stage")

    rows = read_csv(args.data)
    for col in (args.time, args.event):
        if col not in rows[0]:
            sys.exit(f"{args.data}: no column {col!r}")
    usable = [r for r in rows if number(r[args.time]) is not None and parse_event(r[args.event]) is not None]
    max_t = args.limit or max(number(r[args.time]) for r in usable)
    ticks = nice_ticks(max_t)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    limit_txt = f"Follow-up censored at {args.limit:g} months. " if args.limit else ""

    # Cohorts for the stage figure
    stage_cohorts = []
    for spec in args.stage_groups:
        wanted = {ROMAN.get(s.strip().upper()) for s in spec.split(",")}
        if None in wanted:
            sys.exit(f"--stage-groups: cannot read {spec!r}")
        label = "Stage " + "–".join(s.strip().upper() for s in spec.split(","))
        cox_name = "stage " + "-".join(s.strip().upper() for s in spec.split(","))
        stage_cohorts.append((label, cox_name, [r for r in usable if parse_stage(r.get(args.stage_column, "")) in wanted]))

    def panel(x0, y0, title, cohort_rows, feature, cox_row, mark, overall_row=None):
        labels, assign, how = groups_for(cohort_rows, feature, args.split, args.time, args.event, args.limit)
        groups = build_groups(cohort_rows, feature, assign, args.time, args.event, args.limit)
        direction, source = risk_direction(cox_row, overall_row)
        styled_labels, colors = risk_style(labels, direction)
        rename = dict(zip(labels, styled_labels))
        groups = {rename[k]: v for k, v in groups.items()}
        labels = styled_labels
        note = cox_note(cox_row)
        if direction == 0:
            note += "; risk unassigned"
        elif source == "overall":
            note += "; risk from overall HR"
        svg, h, chi2, p = panel_svg(x0, y0, ("● " if mark else "") + title, groups, labels, max_t, ticks,
                                    colors, note=note)
        return svg, h, p, how

    # 1. Overall cohort
    if args.cox:
        model, cox = cox_lookup(read_csv(args.cox), args.model)
        overall = cox.get("All", {})
        chosen = [f for f, r in overall.items() if f in rows[0] and
                  (args.all or (number(r.get("p")) is not None and number(r["p"]) <= args.p))]
        chosen.sort(key=lambda f: number(overall[f].get("p")) if number(overall[f].get("p")) is not None else 2)
        if not chosen:
            print(f"Overall: no feature with Cox p <= {args.p} in {args.cox} ({model})")
        else:
            per_row, parts, y, heights, how = 3, [], 160, [], ""
            for i, f in enumerate(chosen):
                col, row_i = i % per_row, i // per_row
                if col == 0 and i:
                    y += max(heights) + 70
                    heights = []
                mark = number(overall[f].get("p")) is not None and number(overall[f]["p"]) <= 0.05
                svg, h, p, how = panel(16 + col * PANEL_W, y, readable(f, {}, False), usable, f, overall[f], mark)
                parts += svg
                heights.append(h)
            height = y + max(heights) + 24
            width = 16 + min(per_row, len(chosen)) * PANEL_W
            out = args.output_dir / f"{prefix}_all_patients.svg"
            write_svg(out, width, height, f"Overall survival: {what or f'features with Cox p ≤ {args.p:g}'} (all patients)",
                      [f"Groups: {how}. Cox model: {model}; HR per SD. ● = Cox p ≤ 0.05.",
                       f"{limit_txt}Ticks = censored. Lower curve = worse survival.",
                       *RISK_LEGEND], parts)
            print(f"Overall: {len(chosen)} features ({', '.join(chosen)}) -> {out}")

    # 2. Stage-wise
    if args.cox_by_stage:
        model_s, cox_s = cox_lookup(read_csv(args.cox_by_stage), args.model)
        selected = []
        for _, cox_name, _ in stage_cohorts:
            for f, r in cox_s.get(cox_name, {}).items():
                if f in rows[0] and f not in selected and (args.all or (number(r.get("p")) is not None and number(r["p"]) <= args.p)):
                    selected.append(f)
        best_p = {f: min(number(cox_s.get(c, {}).get(f, {}).get("p")) or 1.0 for _, c, _ in stage_cohorts)
                  for f in selected}
        selected.sort(key=lambda f: best_p[f])
        if not selected:
            print(f"Stage-wise: no feature with Cox p <= {args.p} in any stage group of {args.cox_by_stage}")
        else:
            columns = ([] if args.no_all else [("All patients", "All", usable)]) + stage_cohorts
            overall_hr = cox.get("All", {}) if args.cox else {}

            def render(features, out, title):
                parts, y, how = [], 180, ""
                for f in features:
                    parts.append(f'<text x="16" y="{y - 46}" font-size="14" font-weight="600" fill="{INK}">'
                                 f'{escape(readable(f, {}, False))}</text>')
                    heights = []
                    for c, (col_title, cox_name, cohort_rows) in enumerate(columns):
                        cox_row = cox_s.get(cox_name, {}).get(f)
                        if cox_row is None and cox_name == "All":
                            cox_row = overall_hr.get(f)
                        mark = cox_row is not None and number(cox_row.get("p")) is not None and number(cox_row["p"]) <= 0.05
                        svg, h, p, how = panel(16 + c * PANEL_W, y, col_title, cohort_rows, f, cox_row, mark,
                                               overall_hr.get(f))
                        parts += svg
                        heights.append(h)
                    y += max(heights) + 90
                write_svg(out, 16 + len(columns) * PANEL_W, y - 60, title,
                          [f"Groups: {how} within each panel's patients. Cox model: {model_s}; HR per SD. "
                           "● = Cox p ≤ 0.05 in that cohort.",
                           f"{limit_txt}Ticks = censored. Lower curve = worse survival.",
                           *RISK_LEGEND], parts)

            out = args.output_dir / f"{prefix}_by_stage.svg"
            render(selected, out, f"Overall survival by stage: "
                   + (what or f"features with Cox p ≤ {args.p:g} in a stage group"))
            print(f"Stage-wise: {len(selected)} features ({', '.join(selected)}) -> {out}")
            if args.per_feature:
                for f in selected:
                    one = args.output_dir / f"km_feature_{f}.svg"
                    render([f], one, f"Overall survival by {readable(f, {}, False)}: all patients and by stage")
                print(f"Per feature: {len(selected)} files km_feature_<name>.svg in {args.output_dir}")

    if PNG["failures"]:
        sys.exit(f"PNG export failed for {len(PNG['failures'])} plot(s). "
                 "SVGs were saved. Check svg_to_png.py and its rendering dependencies.")


if __name__ == "__main__":
    main()
