#!/usr/bin/env python3
"""Publication-style forest plots (PNG) of Cox regression results.

Example:
    python cox_forest_plots.py cox_univariable.csv cox_adjusted.csv --output-dir forest_plots

Input: CSV files from cox_features.py, or any CSV with the columns
    hr, ci_low, ci_high and feature and/or term
and optionally cohort, model, p, q, n, events, converged, note.

One PNG per input file x cohort x model: <file>__<cohort>__<model>.png.

Rows:
  * Adjusted models (cox_features.py --adjust) store the feature's own term
    plus one row per covariate (AGE, stage II vs I, ...) for every feature.
    Only the feature's own row is plotted (term == feature); --covariates also
    plots the covariate rows, labelled "feature: covariate".
  * Rows without a finite HR / CI, with CI not containing the HR, or with
    converged = False/0/no are skipped (a blank converged counts as converged).
  * Red = p < --alpha (default 0.05), grey = not significant; the HR text of
    significant rows is bold. --significance q uses the BH q-value instead.
  * CIs running past the axis end in an arrow.

Labels: readable names from plot_cox_forest_svg.py when that file is in the
same folder (e.g. log_necrosis_tissue_fraction -> "Necrosis fraction (log)"),
otherwise underscores become spaces. --labels FILE.csv (columns feature,label)
overrides them; --raw-names shows the column names.

Needs: pandas, numpy, matplotlib.
"""

import argparse
import re
import sys
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator  # noqa: E402
from matplotlib.transforms import blended_transform_factory  # noqa: E402

try:  # readable labels shared with the SVG forest plots, if available
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from plot_cox_forest_svg import readable as _svg_readable
except Exception:  # noqa: BLE001
    _svg_readable = None

SIG_COLOR = "#d62828"
NS_COLOR = "#697586"
STRIPE = "#f4f7fa"
TRUE_WORDS = {"true", "t", "1", "yes", "y", ""}
TICKS = [0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50]


def clean_filename(value):
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(value)).strip("._") or "unknown"


def format_p(value):
    if pd.isna(value):
        return "—"
    if value < 0.001:
        return "<0.001"
    return f"{value:.3f}"


def format_hr(row):
    return f"{row['hr']:.2f} ({row['ci_low']:.2f}–{row['ci_high']:.2f})"


def label_for(name, overrides, raw):
    name = str(name)
    if name in overrides:
        return overrides[name]
    if raw:
        return name
    if _svg_readable is not None:
        return _svg_readable(name, {}, False)
    return name.replace("_", " ")


def wrap(text, width=38):
    return "\n".join(textwrap.wrap(str(text), width=width, break_long_words=False,
                                   break_on_hyphens=False)) or str(text)


def is_converged(value):
    if pd.isna(value):
        return True
    return str(value).strip().lower() in TRUE_WORDS


def tick_label(x, _pos=None):
    return f"{x:g}"


def prepare(frame, covariates, overrides, raw):
    """Numeric columns, row labels, and the rows to plot."""
    frame = frame.copy()
    if "feature" not in frame and "term" not in frame:
        raise ValueError("needs a 'feature' or 'term' column")
    missing = [c for c in ("hr", "ci_low", "ci_high") if c not in frame]
    if missing:
        raise ValueError(f"missing column(s): {', '.join(missing)}")
    for c in ("hr", "ci_low", "ci_high", "p", "q", "n", "events"):
        if c in frame:
            frame[c] = pd.to_numeric(frame[c], errors="coerce")
    feature = frame["feature"] if "feature" in frame else frame["term"]
    term = frame["term"] if "term" in frame else frame["feature"]
    feature, term = feature.fillna("").astype(str), term.fillna("").astype(str)
    own = term.eq(feature) | term.eq("")
    if not covariates:
        frame, feature, term, own = frame[own], feature[own], term[own], own[own]
    labels = []
    for f, t, o in zip(feature, term, own):
        name = f if o else t
        text = label_for(name, overrides, raw)
        if not o and f:   # covariate row of an adjusted model: say which model
            text = f"{label_for(f, overrides, raw)}: {text}"
        labels.append(text)
    frame["label"] = labels
    for c, default in (("cohort", "All"), ("model", "")):
        frame[c] = frame[c].fillna(default).astype(str) if c in frame else default
    return frame


def plot_group(group, title, subtitle, destination, dpi=300, sort="input", alpha=0.05,
               significance="p", show_q=False):
    """One forest plot; returns True if a PNG was written."""
    group = group.copy()
    tol = 1e-6   # CIs are rounded in the CSV
    valid = (np.isfinite(group["hr"]) & np.isfinite(group["ci_low"]) &
             np.isfinite(group["ci_high"]) & (group["hr"] > 0) &
             (group["ci_low"] > 0) & (group["ci_high"] > 0) &
             (group["ci_low"] <= group["hr"] * (1 + tol)) &
             (group["hr"] <= group["ci_high"] * (1 + tol)))
    if "converged" in group.columns:
        valid &= group["converged"].map(is_converged)
    skipped = int((~valid).sum())
    if skipped:
        print(f"  Skipping {skipped} invalid/non-converged row(s) in {title}")
    group = group.loc[valid].copy()
    if group.empty:
        print(f"  No valid rows in {title}; no plot created")
        return False

    if sort == "hr":
        group = group.sort_values("hr", kind="stable")
    elif sort == "p" and "p" in group:
        group = group.sort_values("p", na_position="last", kind="stable")
    elif sort == "feature":
        group = group.sort_values("label", kind="stable")
    group = group.reset_index(drop=True)

    nrows = len(group)
    y = np.arange(nrows)[::-1]                       # first row at the top
    labels = [wrap(t) for t in group["label"]]
    nan = pd.Series(np.nan, index=group.index)
    p_values = group["p"] if "p" in group else nan
    q_values = group["q"] if "q" in group else nan
    test = q_values if significance == "q" else p_values
    sig = [bool(pd.notna(v) and v < alpha) for v in test]
    colors = [SIG_COLOR if s else NS_COLOR for s in sig]

    # x range: all CIs, padded, kept within the tick range; longer CIs get arrows
    lo = max(TICKS[0], min(group["ci_low"].min(), 1.0) / 1.25)
    hi = min(TICKS[-1], max(group["ci_high"].max(), 1.0) * 1.25)

    extra_lines = sum(s.count("\n") for s in labels)
    fig_height = max(4.0, 1.9 + 0.43 * nrows + 0.20 * extra_lines)
    fig, ax = plt.subplots(figsize=(13.8, fig_height))
    fig.patch.set_facecolor("white")
    top = 1 - 1.05 / fig_height                       # fixed room for the titles
    bottom = 0.75 / fig_height
    fig.subplots_adjust(left=0.30, right=0.62, top=top, bottom=bottom)

    for k in range(nrows):
        if k % 2 == 0:
            ax.axhspan(y[k] - 0.5, y[k] + 0.5, color=STRIPE, lw=0, zorder=0)
    ax.axvline(1, ls="--", lw=1.3, color="#b14c4c", zorder=1)

    for i, row in group.iterrows():
        c = colors[i]
        a, b = max(row["ci_low"], lo), min(row["ci_high"], hi)
        ax.plot([a, b], [y[i], y[i]], color=c, lw=1.8, solid_capstyle="round", zorder=2)
        for end, clipped, marker in ((a, row["ci_low"] < lo, "<"), (b, row["ci_high"] > hi, ">")):
            if clipped:
                ax.plot(end, y[i], marker=marker, color=c, ms=7, zorder=2)
            else:
                ax.plot([end, end], [y[i] - 0.11, y[i] + 0.11], color=c, lw=1.5, zorder=2)
        ax.scatter(min(max(row["hr"], lo), hi), y[i], s=42, marker="s", color=c,
                   edgecolors="white", linewidths=0.55, zorder=3)

    ax.set_xscale("log")
    ax.set_xlim(lo, hi)
    ax.set_ylim(-0.5, nrows - 0.5)
    ax.xaxis.set_major_locator(FixedLocator([t for t in TICKS if lo <= t <= hi]))
    ax.xaxis.set_major_formatter(FuncFormatter(tick_label))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xlabel("Hazard ratio (95% CI, log scale)", fontsize=10.5)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=10)
    ax.tick_params(axis="y", length=0, pad=8)
    ax.tick_params(axis="x", labelsize=9.5)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.grid(axis="x", color="#e3e8ee", lw=0.8, zorder=0)
    ax.set_axisbelow(True)

    # text columns to the right of the axis: x in axes units, y in data units
    tr = blended_transform_factory(ax.transAxes, ax.transData)
    columns = [(1.05, "HR (95% CI)"), (1.47, "p")] + ([(1.67, "q")] if show_q else [])
    for x, head in columns:
        ax.text(x, nrows - 0.35, head, transform=tr, fontsize=10, fontweight="bold",
                va="bottom", ha="left", clip_on=False)
    for i, row in group.iterrows():
        weight = "bold" if sig[i] else "normal"
        ax.text(1.05, y[i], format_hr(row), transform=tr, fontsize=9.5, va="center",
                fontweight=weight, clip_on=False)
        ax.text(1.47, y[i], format_p(p_values[i]), transform=tr, fontsize=9.5, va="center",
                fontweight=weight if significance == "p" else "normal", clip_on=False)
        if show_q:
            ax.text(1.67, y[i], format_p(q_values[i]), transform=tr, fontsize=9.5,
                    va="center", fontweight=weight if significance == "q" else "normal",
                    clip_on=False)

    # titles in figure units so they never overlap the column headers
    fig.text(0.02, 1 - 0.30 / fig_height, title, fontsize=14, fontweight="bold", va="top")
    n_sig = sum(sig)
    sub = (f"{subtitle}   ·   " if subtitle else "") + \
        f"{n_sig} of {nrows} with {significance} < {alpha:g} (red)   ·   HR > 1 = higher risk"
    fig.text(0.02, 1 - 0.62 / fig_height, sub, fontsize=10, color="#4a5563", va="top")

    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=dpi, facecolor="white")
    plt.close(fig)
    print(f"  Wrote {destination}")
    return True


def counts_text(group):
    parts = []
    for c in ("n", "events"):
        if c in group and group[c].notna().any():
            parts.append(f"{c} = {int(group[c].max())}")
    return ", ".join(parts)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("inputs", nargs="+", type=Path, help="Cox results CSV file(s)")
    parser.add_argument("--output-dir", "--output", "-o", dest="output", type=Path,
                        default=Path("forest_plots"),
                        help="Output folder (default forest_plots; created if missing)")
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--sort", choices=["input", "hr", "p", "feature"], default="input")
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--significance", choices=["p", "q"], default="p",
                        help="Colour red when p (default) or BH q < --alpha")
    parser.add_argument("--show-q", action="store_true", help="Add a q column")
    parser.add_argument("--covariates", action="store_true",
                        help="Also plot covariate rows of adjusted models")
    parser.add_argument("--labels", type=Path, help="CSV with columns feature,label")
    parser.add_argument("--raw-names", action="store_true", help="Show column names")
    args = parser.parse_args()

    overrides = {}
    if args.labels:
        lab = pd.read_csv(args.labels, dtype=str).fillna("")
        overrides = dict(zip(lab["feature"], lab["label"]))

    written = 0
    for path in args.inputs:
        print(path)
        try:
            frame = prepare(pd.read_csv(path), args.covariates, overrides, args.raw_names)
        except (ValueError, OSError, pd.errors.ParserError) as error:
            print(f"  Skipped: {error}")
            continue
        for (cohort, model), group in frame.groupby(["cohort", "model"], sort=False):
            title = f"Cox hazard ratios: {model}" if model else "Cox hazard ratios"
            counts = counts_text(group)
            subtitle = cohort + (f": {counts}" if counts else "")
            name = "__".join(clean_filename(v) for v in (path.stem, cohort, model) if v)
            written += plot_group(group, title, subtitle, args.output / f"{name}.png",
                                  args.dpi, args.sort, args.alpha, args.significance, args.show_q)
    if not written:
        sys.exit("No plots written.")
    print(f"{written} plot(s) in {args.output}")


if __name__ == "__main__":
    main()
