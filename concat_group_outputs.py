#!/usr/bin/env python3
r"""Stack each feature's group files into one CSV per feature.

combine_wsi_outputs.py writes one file per group and feature, e.g.

    out/psg_negative_tumor_core.csv
    out/psg_positive_tumor_core.csv
    out/psg_negative_tils_tsr.csv  ...

This script row-concatenates, for every feature, the files of all groups:

    <output-dir>/all_groups_tumor_core.csv
    <output-dir>/all_groups_tils_tsr.csv  ...

Usage:
    python concat_group_outputs.py --input-dir out out_other_run --output-dir combined
    python concat_group_outputs.py --input-dir out --features tumor_core \
        --groups psg_negative psg_positive --output-dir combined

Files are recognised by name, <group>_<feature>.csv, for the features
tumor_core, immune_proximity, tils_tsr, necrosis_feature and
tumor_morphology (also tumor_morphology_mean / _median). Per-subject files
(TCGA-..._<feature>.csv) and missing_<feature>.csv are skipped.

The 'group' column is kept as written; files without one get it from the file
name. Columns are combined on their union, in first-seen order, with group
first; cells missing in a file are left blank. A subject appearing in more
than one group is reported. Standard library only; values are copied as text.
"""
import argparse
import csv
import re
import sys
from collections import defaultdict
from pathlib import Path

FEATURES = ["tumor_core", "immune_proximity", "tils_tsr", "necrosis_feature",
            "tumor_morphology_mean", "tumor_morphology_median", "tumor_morphology"]
SUBJECT_FILE = re.compile(r"TCGA-[A-Z0-9]{2}-[A-Z0-9]{4}", re.IGNORECASE)


def parse_name(path, features, skip=("missing", "all_groups")):
    """(group, feature) from <group>_<feature>.csv, or None (also for skipped prefixes)."""
    stem = path.stem
    # Longest feature name first, so tumor_morphology_median beats tumor_morphology.
    for feature in sorted(features, key=len, reverse=True):
        if stem.endswith("_" + feature):
            group = stem[: -len(feature) - 1]
            if group and not SUBJECT_FILE.match(group) and group not in skip:
                return group, feature
    return None


def read_csv(path):
    with open(path, newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        header = [c.strip() for c in reader.fieldnames or []]
        rows = [{k.strip(): v for k, v in r.items() if k is not None} for r in reader]
    return header, rows


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input-dir", required=True, nargs="+", type=Path,
                        help="Folder(s) with <group>_<feature>.csv files")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--features", nargs="+",
                        help=f"Features to combine (default: all found): {', '.join(FEATURES)}")
    parser.add_argument("--groups", nargs="+", help="Only these groups (default: all found)")
    parser.add_argument("--prefix", default="all_groups",
                        help="Output file prefix (default: all_groups -> all_groups_<feature>.csv)")
    args = parser.parse_args()

    features = args.features or FEATURES
    found = defaultdict(list)          # feature -> [(group, path)]
    for folder in args.input_dir:
        if not folder.is_dir():
            sys.exit(f"--input-dir is not a folder: {folder}")
        for path in sorted(folder.glob("*.csv")):
            # Earlier combined outputs (prefix) and missing_ lists are not groups.
            parsed = parse_name(path, features, ("missing", "all_groups", args.prefix))
            if parsed is None:
                continue
            group, feature = parsed
            if args.groups and group not in args.groups:
                continue
            found[feature].append((group, path))
    if not found:
        sys.exit("No <group>_<feature>.csv files found in " + ", ".join(map(str, args.input_dir)))
    if args.groups:
        absent = sorted(set(args.groups) - {g for files in found.values() for g, _ in files})
        if absent:
            print(f"Warning: no files for group(s): {absent}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for feature in [f for f in features if f in found]:
        header, rows = ["group"], []
        groups_of = defaultdict(set)   # subject -> groups
        counts = []
        for group, path in found[feature]:
            file_header, file_rows = read_csv(path)
            header += [c for c in file_header if c not in header]
            for row in file_rows:
                if not row.get("group"):
                    row["group"] = group
                groups_of[row.get("subject_id", "")].add(row["group"])
            rows += file_rows
            counts.append(f"{group} {len(file_rows)}")

        out = args.output_dir / f"{args.prefix}_{feature}.csv"
        if out.resolve() in {p.resolve() for _, p in found[feature]}:
            sys.exit(f"{out} would overwrite an input file; choose another --output-dir")
        with open(out, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=header, restval="", extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)

        print(f"\n=== {feature} ===")
        print(f"Groups: {', '.join(counts)} -> {len(rows)} rows, {len(header)} columns")
        shared = sorted(s for s, g in groups_of.items() if s and len(g) > 1)
        if shared:
            print(f"Warning: {len(shared)} subject(s) in more than one group: "
                  + ", ".join(shared[:10]) + (" ..." if len(shared) > 10 else ""))
        print(f"Wrote {out}")


if __name__ == "__main__":
    main()
