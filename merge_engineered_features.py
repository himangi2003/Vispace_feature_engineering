#!/usr/bin/env python3
r"""Combine engineered feature files into one table (one row per subject or WSI).

    python merge_engineered_features.py \
        all_groups_tumor_core_engineered.csv all_groups_tils_tsr_engineered.csv \
        all_groups_immune_proximity_engineered.csv all_groups_necrosis_feature_engineered.csv \
        --output all_groups_engineered_features.csv

Rows are matched on group + subject_id (plus wsi_name when a file has several
rows per subject). Columns present in every file with the same values for the
same row (IDs, clinical and survival columns) are written once, first. All
other columns are features, written after them grouped by input file. A
feature name that appears in two files gets the file's short name as prefix
(e.g. tils_tsr__x); --prefix-all prefixes every feature column.

--how outer (default) keeps every subject found in any file and leaves blanks
where a file has no row; --how inner keeps only subjects present in all files.
A column named in several files with different values for the same subject
stops the merge, unless --allow-conflicts, which keeps the first file's value
and lists the conflicts. Output columns:

    group, subject_id, sample_id, wsi_name, [n_wsi], clinical/survival...,
    <tumor_core features>, <tils_tsr features>, ...

Standard library only.
"""
import argparse
import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path

ID_ORDER = ["group", "subject_id", "sample_id", "wsi_name", "n_wsi"]


def short_name(path):
    stem = Path(path).stem
    for prefix in ("all_groups_",):
        stem = stem[len(prefix):] if stem.startswith(prefix) else stem
    for suffix in ("_engineered",):
        stem = stem[: -len(suffix)] if stem.endswith(suffix) else stem
    return stem


def read(path):
    with open(path, newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        header = [c.strip() for c in reader.fieldnames or []]
        rows = [{k.strip(): (v or "").strip() for k, v in r.items() if k is not None} for r in reader]
    if "subject_id" not in header:
        sys.exit(f"{path}: needs a subject_id column")
    if not rows:
        sys.exit(f"{path}: no data rows")
    return header, rows


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("inputs", nargs="+", type=Path, help="Engineered feature CSVs")
    parser.add_argument("--output", type=Path, default=Path("all_groups_engineered_features.csv"))
    parser.add_argument("--how", choices=["outer", "inner"], default="outer")
    parser.add_argument("--prefix-all", action="store_true",
                        help="Prefix every feature column with its file's short name")
    parser.add_argument("--allow-conflicts", action="store_true",
                        help="Keep the first file's value when shared columns disagree")
    args = parser.parse_args()
    if len(args.inputs) < 2:
        parser.error("give at least two files")
    if args.output.resolve() in {p.resolve() for p in args.inputs}:
        parser.error("--output must not overwrite an input")

    tables = {}
    for path in args.inputs:
        if not path.is_file():
            sys.exit(f"Input not found: {path}")
        name = short_name(path)
        if name in tables:
            sys.exit(f"Two inputs share the name {name!r}")
        tables[name] = (path,) + read(path)

    # Key: group + subject_id, plus wsi_name if any file has several rows per subject.
    key_cols = [c for c in ("group", "subject_id") if all(c in h for _, h, _ in tables.values())]
    per_wsi = any(len({tuple(r[c] for c in key_cols) for r in rows}) < len(rows) for _, _, rows in tables.values())
    if per_wsi:
        if not all("wsi_name" in h for _, h, _ in tables.values()):
            sys.exit("Some file has several rows per subject but not every file has wsi_name")
        key_cols.append("wsi_name")
    for name, (path, _, rows) in tables.items():
        dupes = [k for k, n in Counter(tuple(r[c] for c in key_cols) for r in rows).items() if n > 1]
        if dupes:
            sys.exit(f"{path}: duplicate rows for {key_cols} = {dupes[0]}")
    indexed = {name: {tuple(r[c] for c in key_cols): r for r in rows} for name, (_, _, rows) in tables.items()}

    key_sets = [set(idx) for idx in indexed.values()]
    if args.how == "inner":
        keys = set.intersection(*key_sets)
    else:
        keys = set.union(*key_sets)
    ordered_keys = sorted(keys)

    # Shared columns: in every file (IDs, clinical, survival).
    headers = {name: h for name, (_, h, _) in tables.items()}
    shared = [c for c in headers[next(iter(tables))] if all(c in h for h in headers.values())]
    conflicts = defaultdict(int)
    for k in ordered_keys:
        for c in shared:
            if c in key_cols:
                continue
            values = {name: idx[k][c] for name, idx in indexed.items() if k in idx and idx[k][c] != ""}
            if len(set(values.values())) > 1:
                conflicts[c] += 1
    if conflicts and not args.allow_conflicts:
        sys.exit("Shared columns disagree between files for the same subject: "
                 + ", ".join(f"{c} ({n} rows)" for c, n in conflicts.items())
                 + "\nCheck the inputs, or rerun with --allow-conflicts to keep the first file's value.")
    shared_order = [c for c in ID_ORDER if c in shared] + [c for c in shared if c not in ID_ORDER]

    # Feature columns, grouped by file; prefix on a name clash (or always).
    owner = Counter(c for h in headers.values() for c in h if c not in shared)
    feature_cols = []   # (output name, file, source column)
    for name in tables:
        for c in headers[name]:
            if c in shared:
                continue
            out = f"{name}__{c}" if args.prefix_all or owner[c] > 1 else c
            feature_cols.append((out, name, c))

    out_header = shared_order + [o for o, _, _ in feature_cols]
    present = Counter()
    with open(args.output, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(out_header)
        for k in ordered_keys:
            row = []
            for c in shared_order:
                value = next((idx[k][c] for idx in indexed.values() if k in idx and idx[k][c] != ""), "")
                row.append(value)
            for _, name, c in feature_cols:
                row.append(indexed[name][k][c] if k in indexed[name] else "")
            for name, idx in indexed.items():
                present[name] += k in idx
            writer.writerow(row)

    print(f"Matched on {' + '.join(key_cols)}; {args.how} join -> {len(ordered_keys)} rows, {len(out_header)} columns")
    for name, (path, _, rows) in tables.items():
        n_feat = sum(1 for _, f, _ in feature_cols if f == name)
        print(f"  {name:22} {len(rows):5} rows in file, {present[name]:5} in output, {n_feat:3} feature columns")
    print(f"Shared columns written once ({len(shared_order)}): {', '.join(shared_order)}")
    clashes = sorted({c for _, _, c in feature_cols if owner[c] > 1})
    if clashes and not args.prefix_all:
        print(f"Prefixed because the name is in several files: {', '.join(clashes)}")
    if conflicts:
        print("Warning: shared columns disagreed (first file's value kept): "
              + ", ".join(f"{c} ({n} rows)" for c, n in conflicts.items()))
    missing = len(ordered_keys) - min(present.values())
    if args.how == "outer" and missing:
        print(f"Note: {missing} rows lack at least one feature set (blank cells); --how inner drops them")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
