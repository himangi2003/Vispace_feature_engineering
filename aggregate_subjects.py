#!/usr/bin/env python3
r"""Aggregate per-WSI feature tables to one row per subject.

Input: one or more CSVs with one row per WSI, such as the all_groups_<feature>.csv
files from concat_group_outputs.py (tumor_core, tils_tsr, immune_proximity,
necrosis_feature) or <group>_<feature>.csv from combine_wsi_outputs.py:

    group, subject_id, sample_id, wsi_name, <features...>, <clinical...>

Output, for each input: one row per subject (per group and subject_id when a
group column exists):

    group, subject_id, sample_id, wsi_name, n_wsi, <features...>, <clinical...>

  * Numeric features: mean or median over the subject's WSIs (--method,
    default median; --method-for FEATURE=mean|median overrides it for single
    features, wildcards allowed). Blank and NA cells are ignored; no values -> blank.
  * 0/1 features (one-hot categories, *_missing flags): --binary mode (default,
    most frequent value), any (1 if any WSI is 1) or mean (share of WSIs).
  * Text features (e.g. WSI_TSR_category, wsi_dominant_immune_phenotype):
    the most frequent value over the subject's WSIs.
  * Clinical / survival columns (AGE, OS_MONTHS, OS_STATUS, stage, sex, ...):
    copied; if they differ between a subject's WSIs, the most frequent value
    is kept and a warning is printed.
  * sample_id and wsi_name list the subject's values joined by ";";
    n_wsi counts the WSIs.

Usage:
    # several feature files in one run; each gets <name>_<method>.csv
    python aggregate_subjects.py all_groups_tils_tsr.csv all_groups_immune_proximity.csv \
        --method median --output-dir by_subject

    # one file per group from a combined file (or pass each group's file)
    python aggregate_subjects.py all_groups_necrosis_feature.csv --split-groups \
        --method median --output-dir by_subject
    -> by_subject/<group>_necrosis_feature_median.csv for every group

    # the combined engineered file; area features by mean, the rest by median
    python aggregate_subjects.py all_groups_engineered_features.csv --method median \
        --method-for "log_*area_mm2=mean" --binary any --output-dir by_subject

    # one file, chosen features and output name
    python aggregate_subjects.py all_groups_tumor_core.csv --method mean \
        --features "tumor_*" --output tumor_core_by_subject.csv

Clinical/survival columns are recognised by name; --patient-columns adds
others that must be copied rather than aggregated. --features (wildcards
allowed) restricts the aggregated features; other columns are then copied.
Standard library only.
"""
import argparse
import csv
import fnmatch
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

IDS = ["group", "subject_id", "sample_id", "wsi_name"]
METHODS = {"mean": statistics.fmean, "median": statistics.median}
MISSING = {"", "na", "nan", "none", "null"}
# Patient-level (clinical/survival) column names, case-insensitive wildcards.
PATIENT_PATTERNS = ["age", "diagnosis age", "os_*", "os status", "dfs_*", "pfs_*",
                    "dss_*", "*months*", "*status*", "*days*", "*years*", "*stage*",
                    "*event*", "sex", "gender", "vital_status", "*survival*",
                    "tcga_case_id", "patient_id", "case_id"]


def is_missing(value):
    return value is None or value.strip().lower() in MISSING


def number(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def matches(column, patterns):
    return any(fnmatch.fnmatchcase(column.lower(), p.lower()) for p in patterns)


def most_frequent(values):
    filled = [v for v in values if not is_missing(v)]
    return Counter(filled).most_common(1)[0][0] if filled else ""


def classify(header, rows, features=None, patient_columns=()):
    """Split non-ID columns into numeric features, text features and patient-level."""
    others = [c for c in header if c not in IDS]
    numeric = {c for c in others
               if any(not is_missing(r[c]) for r in rows)
               and all(is_missing(r[c]) or number(r[c]) is not None for r in rows)}
    patient_names = PATIENT_PATTERNS + list(patient_columns)
    if features:
        unknown = [p for p in features if not any(matches(c, [p]) for c in others)]
        if unknown:
            sys.exit(f"--features not found: {unknown}")
        chosen = [c for c in others if matches(c, features) and not matches(c, patient_columns)]
    else:
        chosen = [c for c in others if not matches(c, patient_names)]
    num = [c for c in chosen if c in numeric]
    text = [c for c in chosen if c not in numeric]
    patient = [c for c in others if c not in chosen]
    return others, num, text, patient


def group_file_name(path, group, method):
    """all_groups_<feature>.csv -> <group>_<feature>_<method>.csv."""
    stem = Path(path).stem
    base = stem[len("all_groups_"):] if stem.startswith("all_groups_") else stem
    safe = "".join(ch if ch.isalnum() or ch in "._+-" else "_" for ch in group).strip("_")
    return f"{safe or 'no_group'}_{base}_{method}.csv"


def write_rows(output, header, rows, inputs):
    if output.resolve() in inputs:
        sys.exit(f"{output} would overwrite an input file")
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=header, restval="")
        writer.writeheader()
        writer.writerows(rows)


def aggregate_file(path, method, output, features=None, patient_columns=(),
                   split_groups=False, inputs=frozenset(), method_for=(), binary="mode"):
    with open(path, newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        header = [c.strip() for c in reader.fieldnames or []]
        rows = [{k.strip(): (v or "") for k, v in r.items() if k is not None} for r in reader]
    if "subject_id" not in header:
        sys.exit(f"{path}: needs a subject_id column")
    if not rows:
        sys.exit(f"{path}: no data rows")
    others, num, text, patient = classify(header, rows, features, patient_columns)

    key = ["group", "subject_id"] if "group" in header else ["subject_id"]
    by_subject = defaultdict(list)
    for row in rows:
        by_subject[tuple(row.get(k, "").strip() for k in key)].append(row)

    overrides = {}
    for pair in method_for:
        pattern, _, how = pair.partition("=")
        if how not in METHODS:
            sys.exit(f"--method-for expects FEATURE=mean|median, got {pair!r}")
        hits = [c for c in num if matches(c, [pattern])]
        if not hits:
            sys.exit(f"--method-for: no numeric feature matches {pattern!r}")
        overrides.update({c: how for c in hits})
    is_binary = {c for c in num if {r[c].strip() for r in rows if not is_missing(r[c])} <= {"0", "1"}}
    used = Counter()
    differing = Counter()
    out_rows = []
    for subject_key, subject_rows in by_subject.items():
        out = dict(zip(key, subject_key))
        out["sample_id"] = ";".join(dict.fromkeys(r["sample_id"] for r in subject_rows
                                                  if r.get("sample_id")))
        out["wsi_name"] = ";".join(r["wsi_name"] for r in subject_rows if r.get("wsi_name"))
        out["n_wsi"] = str(len(subject_rows))
        for column in num:
            values = [number(r[column]) for r in subject_rows if not is_missing(r[column])]
            if not values:
                out[column] = ""
            elif len(set(values)) == 1:  # identical: keep the original text
                out[column] = next(r[column] for r in subject_rows if not is_missing(r[column]))
            elif column in is_binary and binary == "mode":
                out[column] = most_frequent(r[column].strip() for r in subject_rows)
            elif column in is_binary and binary == "any":
                out[column] = "1" if 1.0 in values else "0"
            else:
                out[column] = repr(METHODS[overrides.get(column, method)](values))
        for column in text:
            out[column] = most_frequent(r[column] for r in subject_rows)
        for column in patient:
            values = [r[column] for r in subject_rows]
            if len(set(values)) > 1:
                differing[column] += 1
                out[column] = most_frequent(values)
            else:
                out[column] = values[0]
        out_rows.append(out)

    out_header = [c for c in IDS if c in key or c in ("sample_id", "wsi_name")] + ["n_wsi"] + others
    written = []
    if split_groups:
        if "group" not in header:
            sys.exit(f"{path}: --split-groups needs a group column")
        per_group = defaultdict(list)
        for row in out_rows:
            per_group[row["group"]].append(row)
        for group, group_rows in per_group.items():
            target = output.parent / group_file_name(path, group, method)
            write_rows(target, out_header, group_rows, inputs)
            written.append(f"{target} ({len(group_rows)} subjects)")
    else:
        write_rows(output, out_header, out_rows, inputs)
        written.append(str(output))

    multi = sum(len(v) > 1 for v in by_subject.values())
    print(f"\n=== {Path(path).name} ===")
    print(f"{len(rows)} WSI rows -> {len(out_rows)} subjects ({multi} with more than one WSI)")
    continuous = [c for c in num if c not in is_binary]
    by_method = Counter(overrides.get(c, method) for c in continuous)
    print(f"Numeric features ({len(continuous)}): " + ", ".join(f"{n} by {m}" for m, n in by_method.items()))
    if overrides:
        print("  per-feature choice: " + ", ".join(f"{c}={m}" for c, m in overrides.items()))
    if is_binary:
        print(f"0/1 features, {binary} ({len(is_binary)}): {', '.join(sorted(is_binary))}")
    if text:
        print(f"Text features, most frequent value ({len(text)}): {', '.join(text)}")
    print(f"Clinical/survival, copied ({len(patient)}): {', '.join(patient)}")
    if differing:
        print("Warning: clinical/survival columns differ between a subject's WSIs (most "
              "frequent value kept): " + ", ".join(f"{c} ({n})" for c, n in differing.items()))
    if multi == 0:
        print("Note: every subject has one WSI, so the values are unchanged.")
    for line in written:
        print(f"Wrote {line}")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("inputs", nargs="+", type=Path, help="Per-WSI CSV file(s)")
    parser.add_argument("--method", choices=sorted(METHODS), default="median",
                        help="Default aggregation for numeric features (default median)")
    parser.add_argument("--method-for", nargs="+", default=[], metavar="FEATURE=METHOD",
                        help="Per-feature choice, e.g. --method-for log_tumor_area_mm2=mean 'til_*=median'")
    parser.add_argument("--binary", choices=["mode", "any", "mean"], default="mode",
                        help="0/1 features: most frequent value (default), any WSI = 1, or share of WSIs")
    parser.add_argument("--features", nargs="+",
                        help="Feature columns to aggregate, names or wildcards "
                             "(default: every column that is not clinical/survival)")
    parser.add_argument("--patient-columns", nargs="+", default=[],
                        help="Extra columns that are patient-level and copied, not aggregated")
    parser.add_argument("--output-dir", type=Path,
                        help="Folder for <input name>_<method>.csv (default: next to each input)")
    parser.add_argument("--output", type=Path, help="Output file name (only with one input)")
    parser.add_argument("--split-groups", action="store_true",
                        help="Write one file per group: <group>_<feature>_<method>.csv")
    args = parser.parse_args()
    if args.output and len(args.inputs) > 1:
        parser.error("--output works with one input; use --output-dir for several")

    for path in args.inputs:
        if not path.is_file():
            sys.exit(f"Input not found: {path}")
        name = f"{path.stem}_{args.method}.csv"
        output = args.output or (args.output_dir / name if args.output_dir else path.with_name(name))
        aggregate_file(path, args.method, output, args.features, args.patient_columns,
                       args.split_groups, {p.resolve() for p in args.inputs},
                       args.method_for, args.binary)


if __name__ == "__main__":
    main()
