#!/usr/bin/env python3
r"""Feature engineering for tils_tsr features (one row per subject or WSI).

    python tils_tsr_feature_engineering.py all_groups_tils_tsr.csv --save-params tils_tsr_params.json

Recipe (from the distribution review of all_groups_tils_tsr.csv):

    feature                     transform   output column                skew before -> after
    tsr_tumor_to_stroma_ratio   log         log_tsr_ratio                11.0 -> 0.6
    til_stromal_pct             sqrt        sqrt_til_stromal_pct          1.3 -> 0.3
    til_intratumoral_pct        log1p       log_til_intratumoral_pct      1.8 -> -0.3

Left out by default (options add them back):
    tsr_category            identical to tsr_tumor_to_stroma_ratio < 1 (stroma-high);
                            --tsr-flag adds tsr_stroma_high 0/1
    til_between_tumor_pct   rho 0.82 with til_stromal_pct and 45 of 213 missing;
                            --keep-between adds log_til_between_tumor_pct + missing flag

Then each continuous engineered column is:
    1. capped at the --winsorize percentiles (default 1st/99th; 0 = off)
    2. missing values filled with the column median (--impute median, default)
       or left blank (--impute none); a <name>_missing 0/1 flag is added for
       columns that had missing values
    3. standardised to a z-score (--no-standardize skips this)
0/1 columns (flags, one-hot categories) are not capped or scaled.

ID, group and clinical/survival columns pass through unchanged. Fitted values
are saved with --save-params and re-applied to new data with --params (nothing
is re-fitted), e.g. fit on a training set and apply to a test set:

    python tils_tsr_feature_engineering.py train.csv --output train_fe.csv --save-params tils_tsr_params.json
    python tils_tsr_feature_engineering.py test.csv  --output test_fe.csv  --params tils_tsr_params.json

--fit-group fits the parameters on one group only; --rename yours=recipe maps
your column names onto the recipe names. Standard library only.
"""
FEATURE_SET = "tils_tsr"
# (source column, transform, output column)
RECIPE = [
    ("tsr_tumor_to_stroma_ratio", "log", "log_tsr_ratio"),
    ("til_stromal_pct", "sqrt", "sqrt_til_stromal_pct"),
    ("til_intratumoral_pct", "log1p", "log_til_intratumoral_pct"),
]
FLAGS = []      # (source column, value meaning 1, output column)
ONE_HOT = []    # (source column, minimum rows for its own 0/1 column)
DROPPED = []
# option name -> (help, entries); entry = recipe tuple or (source, "flag", value, output)
OPTIONAL = {
    "keep-between": ("Keep til_between_tumor_pct (log1p, with missing flag)",
                     [("til_between_tumor_pct", "log1p", "log_til_between_tumor_pct")]),
    "tsr-flag": ("Add tsr_stroma_high 0/1 (same as tsr_tumor_to_stroma_ratio < 1)",
                 [("tsr_category", "flag", "stroma-high", "tsr_stroma_high")]),
}

# ----------------------------------------------------------------------------
# Engine (identical in every *_feature_engineering.py except tumor_core's)
# ----------------------------------------------------------------------------
import argparse
import csv
import json
import math
import statistics as st
import sys
from collections import Counter
from pathlib import Path

MISSING_TEXT = {"", "na", "nan", "none", "null"}
ID_COLUMNS = {"group", "subject_id", "sample_id", "wsi_name", "n_wsi"}


def number(value):
    if value is None or value.strip().lower() in MISSING_TEXT:
        return None
    try:
        x = float(value)
    except ValueError:
        return None
    return x if math.isfinite(x) else None


def fit_transform_params(values, how):
    """Parameters a transform needs from the fitting data (saved with --save-params)."""
    if how in ("logit", "log_eps"):
        positive = [v for v in values if v is not None and v > 0]
        return {"eps": (min(positive) / 2) if positive else 1e-4}
    return {}


def transform(value, how, p):
    if value is None:
        return None
    if how == "none":
        return value
    if how == "log1p":
        return math.log1p(value) if value > -1 else None
    if how == "log":
        return math.log(value) if value > 0 else None
    if how == "log_eps":
        return math.log(max(value, 0) + p["eps"])
    if how == "sqrt":
        return math.sqrt(value) if value >= 0 else None
    if how == "logit":
        x = min(max(value, 0.0), 1.0)
        return math.log((x + p["eps"]) / (1 - x + p["eps"]))
    if how == "sqrt_beyond_100":
        return math.sqrt(max(100.0 - value, 0.0))
    raise ValueError(how)


def percentile(values, q):
    values = sorted(values)
    pos = (len(values) - 1) * q
    lo, hi = math.floor(pos), math.ceil(pos)
    return values[lo] + (values[hi] - values[lo]) * (pos - lo)


def skew(x):
    if len(x) < 3:
        return float("nan")
    m, s = st.fmean(x), st.pstdev(x)
    return 0.0 if s == 0 else sum((v - m) ** 3 for v in x) / len(x) / s ** 3


def safe_name(text):
    return "".join(ch if ch.isalnum() else "_" for ch in text.lower()).strip("_")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", type=Path, help=f"{FEATURE_SET} CSV")
    parser.add_argument("--output", type=Path, help="Output CSV (default: <input>_engineered.csv)")
    for flag, (help_text, _entries) in OPTIONAL.items():
        parser.add_argument(f"--{flag}", action="store_true", help=help_text)
    parser.add_argument("--winsorize", type=float, default=0.01,
                        help="Cap at this lower/upper percentile (default 0.01; 0 = off)")
    parser.add_argument("--impute", choices=["median", "none"], default="median")
    parser.add_argument("--no-standardize", action="store_true", help="Skip the z-scores")
    parser.add_argument("--fit-group", help="Fit parameters on this group only")
    parser.add_argument("--save-params", type=Path, help="Write the fitted parameters (JSON)")
    parser.add_argument("--params", type=Path, help="Apply saved parameters instead of fitting")
    parser.add_argument("--rename", nargs="+", metavar="YOURS=RECIPE",
                        help="Map your column names to the recipe names")
    args = parser.parse_args()
    if not 0 <= args.winsorize < 0.5:
        parser.error("--winsorize must be between 0 and 0.5")

    with open(args.input, newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        raw_header = [c.strip() for c in reader.fieldnames or []]
        raw_rows = [{k.strip(): (v or "") for k, v in r.items() if k is not None} for r in reader]
    if not raw_rows:
        sys.exit(f"{args.input}: no data rows")

    known = ({s for s, _, _ in RECIPE} | {s for s, _, _ in FLAGS} | {s for s, _ in ONE_HOT}
             | set(DROPPED) | {e[0] for _, es in OPTIONAL.values() for e in es})
    mapping = {}
    for pair in args.rename or []:
        if "=" not in pair:
            sys.exit(f"--rename expects yours=recipe, got {pair!r}")
        yours, recipe_name = (p.strip() for p in pair.split("=", 1))
        if yours not in raw_header:
            sys.exit(f"--rename: column {yours!r} not in the input")
        if recipe_name not in known:
            sys.exit(f"--rename: {recipe_name!r} is not a {FEATURE_SET} recipe column")
        mapping[yours] = recipe_name
    header = [mapping.get(c, c) for c in raw_header]
    rows = [{mapping.get(k, k): v for k, v in r.items()} for r in raw_rows]

    saved = json.loads(args.params.read_text()) if args.params else None
    options = saved["options"] if saved else {flag: getattr(args, flag.replace("-", "_")) for flag in OPTIONAL}
    winsor = saved["winsorize"] if saved else args.winsorize
    impute = saved["impute"] if saved else args.impute
    standardize = saved["standardize"] if saved else not args.no_standardize

    recipe, flags_recipe = list(RECIPE), list(FLAGS)
    for flag, (_help, entries) in OPTIONAL.items():
        if options.get(flag):
            for entry in entries:
                (flags_recipe if entry[1] == "flag" else recipe).append(
                    (entry[0], entry[2], entry[3]) if entry[1] == "flag" else entry)
    one_hot = list(ONE_HOT)
    needed = {s for s, _, _ in recipe} | {s for s, _, _ in flags_recipe} | {s for s, _ in one_hot}
    absent = sorted(needed - set(header))
    if absent:
        others = [c for c in raw_header if c not in ID_COLUMNS]
        sys.exit(f"{args.input}: missing {FEATURE_SET} columns {absent}.\n"
                 f"Columns in the file: {', '.join(others)}\n"
                 "Map yours with --rename yours=recipe_name.")

    fit_rows = rows
    if args.fit_group and not saved:
        if "group" not in header:
            sys.exit("--fit-group needs a group column")
        fit_rows = [r for r in rows if r.get("group") == args.fit_group]
        if not fit_rows:
            sys.exit(f"--fit-group {args.fit_group!r}: no rows")
    fit_ids = {id(r) for r in fit_rows}

    params = saved["columns"] if saved else {}
    engineered, missing_flags, binary, report = {}, {}, {}, []
    for src, how, out in recipe:
        raw_values = [number(r[src]) for r in rows]
        if saved:
            p = params[out]
        else:
            p = {"transform": fit_transform_params(
                [v for r, v in zip(rows, raw_values) if id(r) in fit_ids], how)}
        values = [transform(v, how, p["transform"]) for v in raw_values]
        if not saved:
            fit_values = [v for r, v in zip(rows, values) if id(r) in fit_ids and v is not None]
            if not fit_values:
                sys.exit(f"{src}: no values to fit")
            p["low"] = percentile(fit_values, winsor) if winsor else None
            p["high"] = percentile(fit_values, 1 - winsor) if winsor else None
        capped = [None if v is None else min(max(v, p["low"] if p["low"] is not None else v),
                                             p["high"] if p["high"] is not None else v)
                  for v in values]
        if not saved:
            fit_capped = [v for r, v in zip(rows, capped) if id(r) in fit_ids and v is not None]
            p.update(median=st.median(fit_capped), mean=st.fmean(fit_capped),
                     sd=st.pstdev(fit_capped) or 1.0, had_missing=any(v is None for v in values))
            params[out] = p
        filled = [p["median"] if v is None and impute == "median" else v for v in capped]
        engineered[out] = [None if v is None else (v - p["mean"]) / p["sd"] for v in filled] \
            if standardize else filled
        if p["had_missing"]:
            missing_flags[f"{out}_missing"] = ["1" if v is None else "0" for v in values]
        report.append((src, out, how, skew([v for v in raw_values if v is not None]),
                       skew([v for v in engineered[out] if v is not None]), sum(v is None for v in values)))

    for src, one, out in flags_recipe:
        values = [r[src].strip() for r in rows]
        levels = sorted({v for v in values if v.lower() not in MISSING_TEXT})
        if one not in levels:
            sys.exit(f"{src}: expected the value {one!r}, found {levels}")
        binary[out] = ["" if v.lower() in MISSING_TEXT else ("1" if v == one else "0") for v in values]
        report.append((src, out, f"1 = {one}", float("nan"), float("nan"), sum(v == "" for v in binary[out])))

    for src, min_count in one_hot:
        values = [r[src].strip() for r in rows]
        if saved:
            levels, reference = params[f"_onehot_{src}"]["levels"], params[f"_onehot_{src}"]["reference"]
        else:
            counts = Counter(v for r, v in zip(rows, values) if id(r) in fit_ids and v.lower() not in MISSING_TEXT)
            reference = counts.most_common(1)[0][0]
            levels = [lv for lv, c in counts.most_common() if lv != reference and c >= min_count]
            params[f"_onehot_{src}"] = {"levels": levels, "reference": reference,
                                        "rare": sorted(lv for lv, c in counts.items() if c < min_count and lv != reference)}
        for lv in levels:
            out = f"{safe_name(src)}_{safe_name(lv)}"
            binary[out] = ["" if v.lower() in MISSING_TEXT else ("1" if v == lv else "0") for v in values]
        rare = params[f"_onehot_{src}"]["rare"]
        report.append((src, ", ".join(f"{safe_name(src)}_{safe_name(lv)}" for lv in levels) or "(none)",
                       f"0/1 per category; reference {reference}" + (f"; rare {rare} -> all 0" if rare else ""),
                       float("nan"), float("nan"), 0))

    keep = [c for c in header if c not in known]
    out_names = {c: next((y for y, rc in mapping.items() if rc == c), c) for c in keep}
    out_header = [out_names[c] for c in keep] + list(engineered) + list(binary) + list(missing_flags)
    output = args.output or args.input.with_name(f"{args.input.stem}_engineered.csv")
    if output.resolve() == args.input.resolve():
        sys.exit("--output must not overwrite the input")
    with open(output, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(out_header)
        for i, row in enumerate(rows):
            writer.writerow([row[c] for c in keep]
                            + ["" if engineered[n][i] is None else repr(round(engineered[n][i], 10)) for n in engineered]
                            + [binary[n][i] for n in binary] + [missing_flags[n][i] for n in missing_flags])

    if args.save_params and not saved:
        args.save_params.write_text(json.dumps(dict(
            feature_set=FEATURE_SET, options=options, winsorize=winsor, impute=impute,
            standardize=standardize, fit_group=args.fit_group, columns=params), indent=2))

    print(f"{args.input.name}: {len(rows)} rows; parameters "
          + (f"from {args.params}" if saved else
             f"fitted on {len(fit_rows)} rows" + (f" (group {args.fit_group})" if args.fit_group else "")))
    print(f"{'source':32} {'output':34} {'transform':18} {'skew in':>8} {'skew out':>8} {'missing':>7}")
    for src, out, how, before, after, miss in report:
        print(f"{src:32} {out:34} {how:18} {before:8.2f} {after:8.2f} {miss:7}")
    dropped = sorted((known - needed) & set(header))
    if dropped:
        print("Dropped: " + ", ".join(dropped))
    passed = [out_names[c] for c in keep if c not in ID_COLUMNS]
    print("Passed through unchanged (check that only clinical/survival columns are here): "
          + (", ".join(passed) or "none"))
    print(("z-scored" if standardize else "not standardised")
          + (f"; capped at {winsor:.0%}/{1 - winsor:.0%}" if winsor else "; no capping")
          + ("; missing -> median + flag" if impute == "median" else "; missing left blank")
          + "; 0/1 columns are not scaled")
    print(f"Wrote {output}" + (f" and {args.save_params}" if args.save_params and not saved else ""))


if __name__ == "__main__":
    main()
