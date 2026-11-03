#!/usr/bin/env python3
r"""Feature engineering for tumor_core features (one row per subject or WSI).

Input: a tumor_core table such as all_groups_tumor_core.csv or its aggregated
version from aggregate_subjects.py. ID, group and clinical/survival columns are
passed through unchanged; the tumor features are replaced by engineered ones.

Default recipe (from the distribution review):

    feature                        transform          output column
    tumor_area_mm2                 log1p              log_tumor_area_mm2   (size covariate)
    tumor_area_fraction            none               tumor_area_fraction
    tumor_edge_per_area_per_mm     log1p              log_tumor_edge_per_area_per_mm
    tumor_largest_island_share     none               tumor_largest_island_share
    tumor_island_roundness         sqrt               sqrt_tumor_island_roundness
    tumor_island_solidity          none               tumor_island_solidity
    tumor_island_elongation        log(x - 1)         log_elongation_minus1
    tumor_edge_irregularity        none + impute      tumor_edge_irregularity (+ _missing flag)
    tumor_island_spacing_um        log1p              log_tumor_island_spacing_um
    tumor_island_gap_um            none               tumor_island_gap_um

Dropped as redundant (Spearman rho 0.83-0.96 with tumor_area_mm2 or with
tumor_edge_per_area_per_mm): tumor_region_count, tumor_island_count,
analyzed_area_mm2, tumor_boundary_length_mm, tumor_islands_per_mm2.
--keep-dropped keeps them, log1p-transformed.

--fragmentation chooses how fragmentation enters:
    edge     log_tumor_edge_per_area_per_mm (default)
    islands  log_tumor_islands_per_mm2 instead
    score    fragmentation_score = mean of z(log edge per area), z(log islands
             per mm2) and -z(tumor_area_fraction); the three inputs are dropped

Then, in this order, each engineered column is:
    1. capped at the --winsorize percentiles (default 1st/99th; 0 = off)
    2. missing values filled with the column median (--impute median, default)
       or left blank (--impute none); a <name>_missing 0/1 flag is added for
       columns that had missing values
    3. standardised to a z-score (mean 0, SD 1); --no-standardize skips this

All fitted values (caps, medians, means, SDs) are written to a JSON file.
Apply exactly the same transform to new data (e.g. a test cohort) with
--params, so nothing is re-fitted on it:

    python tumor_core_feature_engineering.py train.csv --output train_fe.csv \
        --save-params tumor_core_params.json
    python tumor_core_feature_engineering.py test.csv --output test_fe.csv \
        --params tumor_core_params.json

--fit-group fits the parameters on one group only (e.g. a reference group)
and applies them to all rows. Standard library only.
"""
import argparse
import csv
import json
import math
import statistics as st
import sys
from pathlib import Path

MISSING = {"", "na", "nan", "none", "null"}

# (source column, transform, output column)
RECIPE = [
    ("tumor_area_mm2", "log1p", "log_tumor_area_mm2"),
    ("tumor_area_fraction", "none", "tumor_area_fraction"),
    ("tumor_edge_per_area_per_mm", "log1p", "log_tumor_edge_per_area_per_mm"),
    ("tumor_largest_island_share", "none", "tumor_largest_island_share"),
    ("tumor_island_roundness", "sqrt", "sqrt_tumor_island_roundness"),
    ("tumor_island_solidity", "none", "tumor_island_solidity"),
    ("tumor_island_elongation", "log_minus1", "log_elongation_minus1"),
    ("tumor_edge_irregularity", "none", "tumor_edge_irregularity"),
    ("tumor_island_spacing_um", "log1p", "log_tumor_island_spacing_um"),
    ("tumor_island_gap_um", "none", "tumor_island_gap_um"),
]
DROPPED = [
    ("tumor_region_count", "log1p", "log_tumor_region_count"),
    ("tumor_island_count", "log1p", "log_tumor_island_count"),
    ("analyzed_area_mm2", "log1p", "log_analyzed_area_mm2"),
    ("tumor_boundary_length_mm", "log1p", "log_tumor_boundary_length_mm"),
    ("tumor_islands_per_mm2", "log1p", "log_tumor_islands_per_mm2"),
]
ALL_FEATURES = {src for src, _, _ in RECIPE + DROPPED}


def number(value):
    if value is None or value.strip().lower() in MISSING:
        return None
    try:
        x = float(value)
    except ValueError:
        return None
    return x if math.isfinite(x) else None


def transform(value, how):
    if value is None:
        return None
    if how == "none":
        return value
    if how == "log1p":
        return math.log1p(value) if value > -1 else None
    if how == "sqrt":
        return math.sqrt(value) if value >= 0 else None
    if how == "log_minus1":
        # elongation is >= 1; values at 1 get a small floor so log is finite
        return math.log(max(value - 1.0, 1e-3))
    raise ValueError(how)


def percentile(values, q):
    values = sorted(values)
    if not values:
        return None
    pos = (len(values) - 1) * q
    lo, hi = math.floor(pos), math.ceil(pos)
    return values[lo] + (values[hi] - values[lo]) * (pos - lo)


def skew(x):
    if len(x) < 3:
        return float("nan")
    m, s = st.fmean(x), st.pstdev(x)
    return 0.0 if s == 0 else sum((v - m) ** 3 for v in x) / len(x) / s ** 3


def build_recipe(fragmentation, keep_dropped):
    recipe = list(RECIPE)
    if fragmentation == "islands":
        recipe = [r for r in recipe if r[0] != "tumor_edge_per_area_per_mm"]
        recipe.insert(2, ("tumor_islands_per_mm2", "log1p", "log_tumor_islands_per_mm2"))
    if keep_dropped:
        recipe += [d for d in DROPPED if d[0] not in {r[0] for r in recipe}]
    return recipe


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", type=Path, help="tumor_core CSV")
    parser.add_argument("--output", type=Path, help="Output CSV (default: <input>_engineered.csv)")
    parser.add_argument("--fragmentation", choices=["edge", "islands", "score"], default="edge")
    parser.add_argument("--keep-dropped", action="store_true",
                        help="Keep the redundant size/density features (log1p)")
    parser.add_argument("--winsorize", type=float, default=0.01,
                        help="Cap at this lower/upper percentile (default 0.01; 0 = off)")
    parser.add_argument("--impute", choices=["median", "none"], default="median")
    parser.add_argument("--no-standardize", action="store_true", help="Skip the z-scores")
    parser.add_argument("--fit-group", help="Fit parameters on this group only")
    parser.add_argument("--save-params", type=Path, help="Write the fitted parameters (JSON)")
    parser.add_argument("--params", type=Path,
                        help="Apply parameters saved earlier instead of fitting (for new data)")
    args = parser.parse_args()
    if not 0 <= args.winsorize < 0.5:
        parser.error("--winsorize must be between 0 and 0.5")

    with open(args.input, newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        header = [c.strip() for c in reader.fieldnames or []]
        rows = [{k.strip(): (v or "") for k, v in r.items() if k is not None} for r in reader]
    if not rows:
        sys.exit(f"{args.input}: no data rows")

    saved = json.loads(args.params.read_text()) if args.params else None
    if saved:
        fragmentation, keep_dropped = saved["fragmentation"], saved["keep_dropped"]
        winsor, impute, standardize = saved["winsorize"], saved["impute"], saved["standardize"]
    else:
        fragmentation, keep_dropped = args.fragmentation, args.keep_dropped
        winsor, impute, standardize = args.winsorize, args.impute, not args.no_standardize

    recipe = build_recipe(fragmentation, keep_dropped)
    needed = {src for src, _, _ in recipe}
    if fragmentation == "score":
        needed |= {"tumor_edge_per_area_per_mm", "tumor_islands_per_mm2", "tumor_area_fraction"}
    absent = sorted(needed - set(header))
    if absent:
        sys.exit(f"{args.input}: missing tumor_core columns {absent}")

    fit_rows = rows
    if args.fit_group and not saved:
        if "group" not in header:
            sys.exit("--fit-group needs a group column")
        fit_rows = [r for r in rows if r.get("group") == args.fit_group]
        if not fit_rows:
            sys.exit(f"--fit-group {args.fit_group!r}: no rows")
    fit_ids = {id(r) for r in fit_rows}

    # 1. transforms
    columns = {}
    for src, how, out in recipe:
        columns[out] = [transform(number(r[src]), how) for r in rows]
    if fragmentation == "score":
        parts = {
            "log_edge": [transform(number(r["tumor_edge_per_area_per_mm"]), "log1p") for r in rows],
            "log_islands": [transform(number(r["tumor_islands_per_mm2"]), "log1p") for r in rows],
            "area_fraction": [number(r["tumor_area_fraction"]) for r in rows],
        }
        for name in ("tumor_area_fraction", "log_tumor_edge_per_area_per_mm"):
            columns.pop(name, None)

    params = saved["columns"] if saved else {}
    report = []

    def fit_and_apply(name, values, final=True):
        fit_values = [v for r, v in zip(rows, values) if id(r) in fit_ids and v is not None]
        p = params.get(name) if saved else {}
        if not saved:
            if not fit_values:
                sys.exit(f"{name}: no values to fit")
            p["low"] = percentile(fit_values, winsor) if winsor else None
            p["high"] = percentile(fit_values, 1 - winsor) if winsor else None
        capped = [None if v is None else
                  min(max(v, p["low"] if p["low"] is not None else v),
                      p["high"] if p["high"] is not None else v) for v in values]
        if not saved:
            fit_capped = [v for r, v in zip(rows, capped) if id(r) in fit_ids and v is not None]
            p["median"] = st.median(fit_capped)
            p["mean"] = st.fmean(fit_capped)
            p["sd"] = st.pstdev(fit_capped) or 1.0
            p["had_missing"] = any(v is None for v in values)
            params[name] = p
        filled = [p["median"] if v is None and impute == "median" else v for v in capped]
        scaled = [None if v is None else (v - p["mean"]) / p["sd"] for v in filled] \
            if standardize and final else filled
        return scaled, capped

    engineered, flags = {}, {}
    for name, values in columns.items():
        engineered[name], capped = fit_and_apply(name, values)
        if params[name]["had_missing"]:
            flags[f"{name}_missing"] = ["1" if v is None else "0" for v in values]
    if fragmentation == "score":
        zs = []
        for name, values in parts.items():
            scaled, _ = fit_and_apply(f"_score_{name}", values, final=False)
            p = params[f"_score_{name}"]
            zs.append([None if v is None else (v - p["mean"]) / p["sd"] for v in scaled])
        score = [None if None in trio else (trio[0] + trio[1] - trio[2]) / 3 for trio in zip(*zs)]
        engineered["fragmentation_score"], _ = fit_and_apply("fragmentation_score", score)

    # Output: every non-feature column, then the engineered features and flags.
    keep = [c for c in header if c not in ALL_FEATURES]
    out_header = keep + list(engineered) + list(flags)
    output = args.output or args.input.with_name(f"{args.input.stem}_engineered.csv")
    if output.resolve() == args.input.resolve():
        sys.exit("--output must not overwrite the input")
    with open(output, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(out_header)
        for i, row in enumerate(rows):
            values = [row[c] for c in keep]
            values += ["" if engineered[n][i] is None else repr(round(engineered[n][i], 10))
                       for n in engineered]
            values += [flags[n][i] for n in flags]
            writer.writerow(values)

    if args.save_params and not saved:
        args.save_params.write_text(json.dumps(dict(
            fragmentation=fragmentation, keep_dropped=keep_dropped, winsorize=winsor,
            impute=impute, standardize=standardize, fit_group=args.fit_group,
            columns=params), indent=2))

    print(f"{args.input.name}: {len(rows)} rows; parameters "
          + (f"from {args.params}" if saved else
             f"fitted on {len(fit_rows)} rows" + (f" (group {args.fit_group})" if args.fit_group else "")))
    print(f"{'source':32} {'output':34} {'skew in':>8} {'skew out':>8} {'missing':>7}")
    sources = {out: src for src, _, out in recipe}
    for name in engineered:
        src = sources.get(name, "(composite)")
        raw = [number(r[src]) for r in rows] if src in header else []
        raw = [v for v in raw if v is not None]
        new = [v for v in engineered[name] if v is not None]
        missing = sum(v is None for v in columns.get(name, engineered[name]))
        print(f"{src:32} {name:34} {skew(raw) if raw else float('nan'):8.2f} {skew(new):8.2f} {missing:7}")
    dropped = sorted(ALL_FEATURES - {src for src, _, _ in recipe}
                     - ({"tumor_edge_per_area_per_mm", "tumor_islands_per_mm2", "tumor_area_fraction"}
                        if fragmentation == "score" else set()))
    if fragmentation == "score":
        print("fragmentation_score built from tumor_edge_per_area_per_mm, tumor_islands_per_mm2, "
              "tumor_area_fraction (inputs not kept)")
    if dropped:
        print("Dropped: " + ", ".join(dropped))
    print(("z-scored" if standardize else "not standardised")
          + (f"; capped at {winsor:.0%}/{1 - winsor:.0%}" if winsor else "; no capping")
          + (f"; missing -> median + flag" if impute == "median" else "; missing left blank"))
    print(f"Wrote {output}" + (f" and {args.save_params}" if args.save_params and not saved else ""))


if __name__ == "__main__":
    main()
