#!/usr/bin/env python3
r"""Combine per-WSI feature outputs with clinical and survival data.

Input layout (searched recursively under --root, so subfolders may differ):

    <root>/<wsi_name>/spatial_feature_results/
        Master ROI/tils_tsr_wsi_summary.csv
        immune_proximity/immune_proximity_wsi_summary.csv
        necrosis_feature/necrosis_feature_wsi_summary.csv
        tumor_morphology/tumor_core_wsi_summary.csv

Features (--features, default: all four):

    tumor_core         tumor_core_wsi_summary.csv
    immune_proximity   immune_proximity_wsi_summary.csv
    tils_tsr           tils_tsr_wsi_summary.csv
    necrosis_feature   necrosis_feature_wsi_summary.csv

Output (one file per feature, one row per WSI):

    one subject         <output-dir>/<subject_id>_<feature>.csv
    list of subjects    <output-dir>/<group-name>_<feature>.csv
                        (merged_<feature>.csv without --group-name)

Output columns, in order:

    group               --group-name (only for a list of subjects)
    subject_id, sample_id, wsi_name
    feature columns     --feature-columns (default: all)
    clinical columns    --clinical-columns (default: all)
    survival columns    --survival-columns (default: OS_MONTHS, OS_STATUS)

Examples:

    # one subject
    python combine_wsi_outputs.py --root /path/outputs \
        --subject TCGA-A7-A13H --features tumor_core \
        --clinical data/master_clinical.csv \
        --clinical-columns "Diagnosis Age" --output-dir out

    # all subjects in an ID file, merged, with a group column
    python combine_wsi_outputs.py --root /path/outputs \
        --subject ilc_ids.txt --group-name ILC --features tumor_core tils_tsr \
        --clinical data/master_clinical.csv \
        --clinical-columns "Diagnosis Age" \
            "Neoplasm Disease Stage American Joint Committee on Cancer Code" \
        --output-dir out

    # list the column names available for selection
    python combine_wsi_outputs.py --root /path/outputs \
        --clinical data/master_clinical.csv --list-columns

Notes:
  * --subject takes ID files (one ID per line) or IDs. An ID may be a patient
    (TCGA-XX-XXXX), sample (TCGA-XX-XXXX-01) or WSI name; blank lines and
    # comments are ignored. Several files without --group-name get each
    file's name as group. --per-subject also writes per-subject files.
  * Survival: with --clinical, OS_MONTHS and OS_STATUS are added by default,
    read from "Overall Survival (Months)" and "Overall Survival Status"
    (0:LIVING / 1:DECEASED). --survival-columns chooses others; with no names
    it adds none. A separate --survival file supplies them instead.
  * Clinical: all columns of --clinical by default, in every feature file;
    --clinical-columns picks some, with no names it adds none. Names accept
    wildcards ("*Stage*", "wsi_tumor_*"); the clinical__ and survival__
    prefixes are removed unless --keep-prefixes is given.
  * Rows are matched to clinical rows on wsi_name (ignoring folder, .svs-type
    extension and case), then sample, then patient; the report shows how.
  * Clinical rows are matched on wsi_name, sample_id or patient ID, whichever
    the clinical file has; unmatched rows keep blank clinical cells.
  * Subjects without output are listed in missing_<feature>.csv.
  * Standard library only; values are copied as text, unchanged.
"""
import argparse
import csv
import fnmatch
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

PATIENT = re.compile(r"TCGA-[A-Z0-9]{2}-[A-Z0-9]{4}", re.IGNORECASE)
SAMPLE = re.compile(r"(TCGA-[A-Z0-9]{2}-[A-Z0-9]{4})-\d{2}", re.IGNORECASE)
WSI = re.compile(r"((TCGA-[A-Z0-9]{2}-[A-Z0-9]{4})-\d{2})[A-Z]?-.+", re.IGNORECASE)

FRONT = ["subject_id", "sample_id", "wsi_name"]

FEATURES = {
    "tumor_core": "tumor_core_wsi_summary.csv",
    "immune_proximity": "immune_proximity_wsi_summary.csv",
    "tils_tsr": "tils_tsr_wsi_summary.csv",
    "necrosis_feature": "necrosis_feature_wsi_summary.csv",
}


def resolve_features(names):
    """[(feature name, summary file name)] from short names or file names."""
    resolved = []
    for name in names:
        if name in FEATURES:
            pair = (name, FEATURES[name])
        elif name.endswith(".csv"):
            short = re.sub(r"(_wsi)?_summary\.csv$|\.csv$", "", name)
            pair = (short, name)
        else:
            sys.exit(f"Unknown feature {name!r}; choose from {', '.join(FEATURES)} "
                     "or give a summary file name ending in .csv")
        if pair not in resolved:
            resolved.append(pair)
    return resolved


def parse_id(value, where):
    """(level, normalised ID) for a patient, sample or WSI ID."""
    if WSI.fullmatch(value):
        return "wsi", value
    if SAMPLE.fullmatch(value):
        return "sample", value.upper()
    if PATIENT.fullmatch(value):
        return "patient", value.upper()
    sys.exit(f"{where}: not a TCGA patient, sample or WSI ID: {value!r}")


def read_ids(paths, labels=None, subjects=None, subject_label=""):
    """IDs from text files and/or given directly (subjects).

    Returns ({level: {id: label}}, [(level, id, label)]) with level patient,
    sample or wsi; label is the file's group name (labels, else file stem).
    """
    labels = labels or [Path(p).stem for p in paths]
    if len(labels) != len(paths):
        sys.exit("--group-names needs one name per --ids file")
    sources = [(f"{path}, line {number}", line, label)
               for path, label in zip(paths, labels)
               for number, line in enumerate(
                   Path(path).read_text(encoding="utf-8-sig").splitlines(), 1)]
    sources += [("--subject", value, subject_label) for value in subjects or []]
    ids = {"patient": {}, "sample": {}, "wsi": {}}
    order = []
    for where, line, label in sources:
        value = line.split("#", 1)[0].strip().strip(",;\"'")
        if not value:
            continue
        level, value = parse_id(value, where)
        if value in ids[level]:
            if ids[level][value] != label:
                print(f"Warning: {value} is in more than one ID file; "
                      f"kept in {ids[level][value]!r}")
            continue
        ids[level][value] = label
        order.append((level, value, label))
    if not order:
        sys.exit("No IDs found in --ids/--subject")
    return ids, order


def find_files(root, summary, subjects=None):
    """Summary files under root. With subjects (patient IDs), look first only in
    folders named after them (<root>/<wsi or sample name>/...); fall back to the
    whole tree when there are none."""
    def search(folder):
        return [p for p in Path(folder).rglob("*" + summary)
                if p.is_file() and (p.name == summary or p.name.endswith("__" + summary))]
    if subjects:
        folders = [f for subject in sorted(subjects) for f in Path(root).glob(subject + "*")
                   if f.is_dir()]
        files = [p for folder in folders for p in search(folder)]
        # Slide folders named TCGA-...: the patient is simply not under this root.
        tcga_layout = any(f.is_dir() and PATIENT.match(f.name) for f in Path(root).iterdir())
        if files or tcga_layout:
            return sorted(set(files))
    return sorted(search(root))


def context_ids(path, root, summary):
    """WSI name and sample ID implied by a file's name and enclosing folders."""
    wsi_name = path.name[: -len("__" + summary)] if path.name != summary else ""
    sample_id = ""
    # Nearest folder first; the --root folder itself counts too.
    folders = (Path(root).name,) + path.relative_to(root).parts[:-1]
    for folder in reversed(folders):
        if not wsi_name and WSI.fullmatch(folder):
            wsi_name = folder
        if not sample_id and SAMPLE.fullmatch(folder):
            sample_id = folder.upper()
    return wsi_name, sample_id


def identify(wsi_name, folder_sample):
    """(subject_id, sample_id) from a WSI name, else from the sample folder."""
    match = WSI.fullmatch(wsi_name or "")
    if match:
        return match.group(2).upper(), match.group(1).upper()
    if folder_sample:
        return SAMPLE.fullmatch(folder_sample).group(1), folder_sample
    return "", ""


def wanted(row, ids):
    """Group label of the first matching ID (most specific level first), else None."""
    for level, column in (("wsi", "wsi_name"), ("sample", "sample_id"), ("patient", "subject_id")):
        if row[column] in ids[level]:
            return ids[level][row[column]]
    return None


def combine(roots, id_files, summary, wsi_column="wsi_name", add_source=False,
            group_names=None, add_group=None, subjects=None, group_name=None):
    """Rows of every WSI under roots whose subject is in id_files (all if none).

    add_group (default: when there is more than one ID file or group_names is
    given) adds a 'group' column holding the ID file's group name.
    """
    roots = [Path(r).resolve() for r in ([roots] if isinstance(roots, (str, Path)) else roots)]
    for root in roots:
        if not root.is_dir():
            sys.exit(f"--root is not a folder: {root}")
    id_files = list(id_files or [])
    subjects = list(subjects or [])
    if group_name and not group_names and id_files:
        group_names = [group_name] * len(id_files)
    ids, order = (read_ids(id_files, group_names, subjects, group_name or "")
                  if id_files or subjects else (None, []))
    if add_group is None:
        add_group = len(id_files) > 1 or bool(group_names) or bool(group_name)
    # With an ID list, search just those patients' folders (fast).
    patients = {PATIENT.match(value).group(0).upper() for _, value, _ in order} or None
    files = [(root, path) for root in roots for path in find_files(root, summary, patients)]
    if not files:
        raise FileNotFoundError(f"No {summary} files found under {', '.join(map(str, roots))}")

    rows, columns, problems = [], [], []
    for root, path in files:
        with open(path, newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            header = [c.strip() for c in (reader.fieldnames or [])]
            if len(set(header)) != len(header):
                problems.append(f"{path}: duplicate column names; skipped")
                continue
            records = [{k.strip(): (v or "") for k, v in r.items() if k is not None} for r in reader]
        if not records:
            problems.append(f"{path}: no data rows; skipped")
            continue
        file_wsi, folder_sample = context_ids(path, root, summary)
        if len(records) > 1 and wsi_column not in header:
            problems.append(f"{path}: {len(records)} rows but no {wsi_column} column; skipped")
            continue
        for record in records:
            wsi_name = clean_wsi(record.get(wsi_column, "")) or file_wsi
            subject_id, sample_id = identify(wsi_name, folder_sample)
            if not subject_id:
                problems.append(f"{path}: cannot identify the subject of a row; skipped")
                continue
            row = {"subject_id": subject_id, "sample_id": sample_id, "wsi_name": wsi_name}
            if ids is not None:
                group = wanted(row, ids)
                if group is None:
                    continue
                if add_group:
                    row["group"] = group
            for column in header:
                if column == wsi_column:
                    continue
                name = column if column not in FRONT else f"source__{column}"
                if column in ("subject_id", "sample_id") and record[column].strip().upper() == row[column]:
                    continue  # identical to the derived ID; no need to keep a copy
                row[name] = record[column]
                if name not in columns:
                    columns.append(name)
            if add_source:
                row["source_file"] = str(path)
            rows.append(row)

    # A WSI must appear once; the same slide in two files is ambiguous.
    counts = Counter(r["wsi_name"] for r in rows if r["wsi_name"])
    duplicated = sorted(w for w, n in counts.items() if n > 1)
    if duplicated:
        sys.exit("Same WSI found in more than one file: " + ", ".join(duplicated[:10])
                 + "\nRemove the stale copies or point --root at one outputs folder.")

    rows.sort(key=lambda r: (r["subject_id"], r["sample_id"], r["wsi_name"]))
    per_subject = Counter(r["subject_id"] for r in rows)
    found = {"wsi": {r["wsi_name"] for r in rows}, "sample": {r["sample_id"] for r in rows},
             "patient": {r["subject_id"] for r in rows}}
    missing = [(value, label) for level, value, label in order if value not in found[level]]

    header = (["group"] if add_group else []) + FRONT + columns \
        + (["source_file"] if add_source else [])
    id_source = ", ".join([Path(f).name for f in id_files] + (["--subject"] if subjects else []))
    id_note = (f"IDs in {id_source}: {len(order)}; "
               f"with output: {len(order) - len(missing)}; without output: {len(missing)}"
               if order else "No --ids given: every subject found is kept")
    report = [
        f"{summary} files found: {len(files)} (under {', '.join(map(str, roots))})",
        id_note,
        f"Rows (WSIs) written: {len(rows)} from {len(per_subject)} subjects; "
        f"subjects with more than one WSI: {sum(n > 1 for n in per_subject.values())}",
    ]
    return header, rows, missing, problems, report


PATIENT_ALIASES = ("subject_id", "patient_id", "tcga_case_id", "Patient ID",
                   "case_id", "bcr_patient_barcode")
PREFIXES = ("clinical__", "survival__")
ID_COLUMNS = ("subject_id", "sample_id", "wsi_name")

# Short survival names -> clinical columns that hold them (cBioPortal export).
SURVIVAL_ALIASES = {
    "OS_MONTHS": ("Overall Survival (Months)", "OS_MONTHS"),
    "OS_STATUS": ("Overall Survival Status", "OS_STATUS"),
}
DEFAULT_SURVIVAL = ["OS_MONTHS", "OS_STATUS"]


# ------------------------------------------------------------------ clinical tables
def read_table(path):
    """Header and rows; duplicate column names get .1, .2 suffixes (as pandas)."""
    with open(path, newline="", encoding="utf-8-sig") as handle:
        reader = csv.reader(handle)
        raw = [c.strip() for c in next(reader, [])]
        header, seen = [], defaultdict(int)
        for column in raw:
            name = column if not seen[column] else f"{column}.{seen[column]}"
            seen[column] += 1
            header.append(name)
        duplicated = sorted(c for c, n in seen.items() if n > 1)
        if duplicated:
            print(f"Note: {path} repeats column(s) {duplicated}; later copies renamed .1, .2")
        rows = [{name: value.strip() for name, value in zip(header, record)}
                for record in reader if any(v.strip() for v in record)]
    return header, rows


def patient_column(header, path):
    for alias in PATIENT_ALIASES:
        for column in header:
            if column == alias or column.split("__", 1)[-1] == alias:
                return column
    sys.exit(f"{path}: no patient ID column (tried {', '.join(PATIENT_ALIASES)})")


def strip_prefixes(header, taken):
    """{old: new} for prefixed columns whose short name is free."""
    short = {}
    for column in header:
        for prefix in PREFIXES:
            if column.startswith(prefix):
                short[column] = column[len(prefix):]
    counts = defaultdict(int)
    for new in short.values():
        counts[new] += 1
    blocked = set(taken) | (set(header) - set(short))
    return {old: new for old, new in short.items() if counts[new] == 1 and new not in blocked}


def clean_wsi(name):
    """WSI name without a folder path or slide-file extension (case kept)."""
    name = str(name or "").replace("\\", "/").rsplit("/", 1)[-1].strip()
    return re.sub(r"\.(svs|tiff?|ndpi|mrxs|scn|vms|bif)$", "", name, flags=re.IGNORECASE)


def wsi_key(name):
    """WSI name without folder, slide-file extension or case differences."""
    return clean_wsi(name).upper()


class Clinical:
    """A clinical table indexed by its most specific ID column."""

    def __init__(self, path, keep_prefixes=False, label="clinical"):
        header, rows = read_table(path)
        if not rows:
            sys.exit(f"{path}: no data rows")
        self.path, self.label = path, label
        patient = "subject_id" if "subject_id" in header else patient_column(header, path)
        if "sample_id" not in header:
            alias = next((c for c in header if c.split("__", 1)[-1] in ("Sample ID", "sample")), None)
            if alias:
                for row in rows:
                    row["sample_id"] = row[alias]
                header = header + ["sample_id"]
        for row in rows:
            match = PATIENT.match(row.get(patient, "") or row.get("sample_id", "")
                                  or row.get("wsi_name", ""))
            row["subject_id"] = match.group(0).upper() if match else ""
            if "sample_id" in row:
                row["sample_id"] = row["sample_id"].upper()
        self.key = next(c for c in ("wsi_name", "sample_id", "subject_id")
                        if c in header or c == "subject_id")
        # Columns to add: everything except IDs and the raw patient column.
        self.columns = [c for c in header if c not in ID_COLUMNS and c != patient]
        self.rename = {} if keep_prefixes else strip_prefixes(self.columns, ())
        self.by_key, conflicts = {}, []
        for row in rows:
            value = row.get(self.key, "")
            if not value:
                continue
            if value in self.by_key and self.by_key[value] != row:
                conflicts.append(value)
            self.by_key.setdefault(value, row)
        if conflicts:
            sys.exit(f"{path}: different rows for the same {self.key}: {conflicts[:5]}")
        # Looser indexes, used when the exact ID does not match: WSI names
        # without folder/extension/case differences, then sample, then patient.
        self.by_wsi = {wsi_key(r["wsi_name"]): r for r in rows if r.get("wsi_name")} \
            if self.key == "wsi_name" else {}
        self.by_sample, self.by_subject = defaultdict(list), defaultdict(list)
        for row in rows:
            if row.get("sample_id"):
                self.by_sample[row["sample_id"]].append(row)
            self.by_subject[row["subject_id"]].append(row)

    def names(self, taken):
        """Output names of the clinical columns, avoiding names in taken."""
        names = {}
        for column in self.columns:
            new = self.rename.get(column, column)
            names[column] = new if new not in taken else column
        return names

    def lookup(self, row):
        """Clinical row for a feature row, and how it was matched.

        Tries the exact ID, then the WSI name ignoring folder, extension and
        case, then the sample, then the patient. When several clinical rows
        share the sample/patient, columns on which they agree are used and
        the others are left blank.
        """
        value = row.get(self.key, "").upper() if self.key != "wsi_name" else row.get("wsi_name", "")
        if value and value in self.by_key:
            return self.by_key[value], self.key
        if self.by_wsi and row.get("wsi_name"):
            match = self.by_wsi.get(wsi_key(row["wsi_name"]))
            if match is not None:
                return match, "wsi_name (normalised)"
        for level, index in (("sample_id", self.by_sample), ("subject_id", self.by_subject)):
            candidates = index.get((row.get(level) or "").upper(), [])
            if len(candidates) == 1:
                return candidates[0], level
            if candidates:
                agreed = {c: candidates[0].get(c, "") for c in self.columns
                          if all(r.get(c, "") == candidates[0].get(c, "") for r in candidates)}
                return agreed, f"{level} (shared values)"
        return None, None


# ------------------------------------------------------------------ output
def write_csv(path, header, rows):
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=header, restval="", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def match_columns(columns, patterns):
    """Columns matching exact names or wildcards, in pattern order; also the hit patterns."""
    chosen, hit = [], set()
    for pattern in patterns:
        if any(ch in pattern for ch in "*?["):
            hits = [c for c in columns if fnmatch.fnmatchcase(c, pattern)]
        else:
            hits = [pattern] if pattern in columns else []
        if hits:
            hit.add(pattern)
        chosen += [h for h in hits if h not in chosen]
    return chosen, hit


def pick(table, patterns, taken, what, default_all):
    """[(source column, output name)] chosen from a clinical table."""
    if table is None:
        return []
    names = table.names(taken)
    if patterns is None:
        # Skip columns already present, also when only the prefixed name is left.
        return [(c, n) for c, n in names.items()
                if n not in taken and table.rename.get(c, c) not in taken] if default_all else []
    output = list(names.values())
    by_name = {n: c for c, n in names.items()}
    chosen, missing = [], []
    for pattern in patterns:
        hits, _ = match_columns(output, [pattern])
        if hits:
            chosen += [(by_name[n], n) for n in hits if (by_name[n], n) not in chosen]
            continue
        # Short survival names (OS_MONTHS, OS_STATUS) read their clinical column.
        alias = next((a for a in SURVIVAL_ALIASES.get(pattern.upper(), ()) if a in by_name), None)
        if alias:
            chosen.append((by_name[alias], pattern.upper()))
        else:
            missing.append(pattern)
    if missing:
        sys.exit(f"{what} not found in {table.path}: {missing}\n"
                 "Run with --list-columns to see the available names.")
    return chosen


def add_clinical(header, rows, clinical, survival, feature_columns=None,
                 clinical_columns=None, survival_columns=None, add_source=False):
    """Keep the chosen feature columns and add clinical/survival columns to rows.

    clinical/survival are Clinical tables (or None).
    Returns the output header (ID, feature, clinical, survival[, source_file])
    and an info dict with the column groups and per-table unmatched row counts.
    """
    front = [c for c in header if c in ("group",) + tuple(FRONT)]
    values = [c for c in header if c not in front and c != "source_file"]
    hit = set()
    if feature_columns:
        values, hit = match_columns(values, feature_columns)
    taken = set(front) | set(values) | {"source_file"}

    # Survival first from its own file; clinical columns then skip those names.
    survival_table = survival or clinical
    survival_cols = pick(survival_table, survival_columns, taken, "Survival columns",
                         default_all=survival is not None and survival_columns is None)
    taken |= {n for _, n in survival_cols}
    clinical_cols = pick(clinical, clinical_columns, taken, "Clinical columns",
                         default_all=clinical_columns is None)
    if survival is None:
        # Same file: a column already added as survival is not repeated.
        used = {source for source, _ in survival_cols}
        clinical_cols = [c for c in clinical_cols if c[0] not in used]

    unmatched, matched_by = defaultdict(int), defaultdict(int)
    for row in rows:
        counted = set()
        for table, chosen in ((clinical, clinical_cols), (survival_table, survival_cols)):
            if not chosen:
                continue
            source, how = table.lookup(row)
            if id(table) not in counted:  # clinical and survival may be one file
                counted.add(id(table))
                matched_by[f"{table.label} by {how}" if how else f"{table.label}: no match"] += 1
                if source is None:
                    unmatched[table.label] += 1
            for column, name in chosen:
                row[name] = source.get(column, "") if source else ""

    clinical_names = [n for _, n in clinical_cols]
    survival_names = [n for _, n in survival_cols]
    header = front + values + clinical_names + survival_names \
        + (["source_file"] if add_source else [])
    return header, {"front": front, "features": values, "clinical": clinical_names,
                    "survival": survival_names, "unmatched": dict(unmatched),
                    "matched_by": dict(matched_by),
                    "feature_patterns_hit": hit}


def column_summary(info):
    text = (f"Columns: {len(info['front'])} ID + {len(info['features'])} feature + "
            f"{len(info['clinical'])} clinical + {len(info['survival'])} survival")
    if info.get("matched_by"):
        text += "\nRow matching: " + ", ".join(f"{k} {v}" for k, v in info["matched_by"].items())
    return text


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", required=True, type=Path, nargs="+",
                        help="One or more outputs folders, searched recursively")
    parser.add_argument("--subject", nargs="+",
                        help="Text file(s) of subject IDs, one per line, whose subjects are "
                             "merged into one file per feature; plain IDs work too, e.g. "
                             "--subject ilc_ids.txt or --subject TCGA-A7-A13H")
    parser.add_argument("--ids", type=Path, nargs="+",
                        help="One or more ID text files (default: keep every subject found)")
    parser.add_argument("--group-names", nargs="+",
                        help="Group name per --ids file for the 'group' column "
                             "(default: file names; the column is added with 2+ files)")
    parser.add_argument("--features", nargs="+", default=list(FEATURES),
                        help=f"Features to process (default: all): {', '.join(FEATURES)}, "
                             "or a summary file name")
    parser.add_argument("--feature-columns", nargs="+",
                        help="Feature columns to keep, names or wildcards, e.g. 'wsi_tumor_*' "
                             "(default: all columns of each summary)")
    parser.add_argument("--clinical", type=Path,
                        help="Clinical CSV (with wsi_name, sample_id or a patient ID column)")
    parser.add_argument("--clinical-columns", nargs="*",
                        help="Clinical columns to add, names or wildcards (default: all "
                             "clinical columns; give the option with no names for none)")
    parser.add_argument("--survival", type=Path,
                        help="Optional per-patient survival CSV; without it survival columns "
                             "are taken from --clinical")
    parser.add_argument("--survival-columns", nargs="*",
                        help="Survival columns to add (default: OS_MONTHS OS_STATUS, read from "
                             "'Overall Survival (Months)' and 'Overall Survival Status'; with "
                             "--survival, default all its columns; give the option with no "
                             "names for none)")
    parser.add_argument("--keep-prefixes", action="store_true",
                        help="Keep clinical__/survival__ prefixes on clinical column names")
    parser.add_argument("--list-columns", action="store_true",
                        help="Print the available feature, clinical and survival columns and stop")
    parser.add_argument("--output-dir", type=Path,
                        help="Folder for <subject_id>_<feature>.csv files (created if needed)")
    parser.add_argument("--group-name",
                        help="Group name for the listed subjects: fills a 'group' column and "
                             "names the merged file <group-name>_<feature>.csv")
    parser.add_argument("--per-subject", action="store_true",
                        help="With several subjects, also write <subject_id>_<feature>.csv each")
    parser.add_argument("--wsi-column", default="wsi_name")
    args = parser.parse_args()
    if not args.output_dir and not args.list_columns:
        parser.error("--output-dir is required")
    # --subject: ID files are read like --ids; anything else is an ID itself.
    subject_files = [Path(v) for v in args.subject or [] if Path(v).is_file()]
    args.subject = [v for v in args.subject or [] if not Path(v).is_file()] or None
    args.ids = (args.ids or []) + subject_files or None

    features = resolve_features(args.features)
    clinical = Clinical(args.clinical, args.keep_prefixes) if args.clinical else None
    survival = Clinical(args.survival, args.keep_prefixes, "survival") if args.survival else None
    if (args.clinical_columns or args.survival_columns) and not (clinical or survival):
        parser.error("--clinical-columns/--survival-columns need --clinical or --survival")
    if args.survival_columns is None:
        # Default survival columns: OS_MONTHS and OS_STATUS from the clinical file;
        # with a separate --survival file, all of its columns.
        args.survival_columns = None if survival else DEFAULT_SURVIVAL if clinical else []

    if args.list_columns:
        for feature, summary in features:
            try:
                header = combine(args.root, args.ids, summary, args.wsi_column, False,
                                 args.group_names, subjects=args.subject,
                group_name=args.group_name)[0]
            except FileNotFoundError as error:
                print(f"\n{feature}: {error}")
                continue
            print(f"\n{feature} columns ({summary}):\n  " + "\n  ".join(header))
        for label, table in (("clinical", clinical), ("survival", survival)):
            if table:
                print(f"\n{label} columns ({table.path}, joined on {table.key}):\n  "
                      + "\n  ".join(table.names(set()).values()))
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_dir = args.output_dir.resolve()
    unused = set(args.feature_columns or [])

    for feature, summary in features:
        print(f"\n=== {feature} ({summary}) ===")
        try:
            header, rows, missing, problems, report = combine(
                args.root, args.ids, summary, args.wsi_column, False,
                args.group_names, subjects=args.subject,
                group_name=args.group_name)
        except FileNotFoundError as error:
            print(f"Skipped: {error}")
            continue

        header, info = add_clinical(header, rows, clinical, survival, args.feature_columns,
                                    args.clinical_columns, args.survival_columns,
                                    False)
        unused -= info["feature_patterns_hit"]

        # One requested subject -> <subject_id>_<feature>.csv; a list of subjects
        # (or all subjects) -> one merged <group-name or merged>_<feature>.csv.
        single = not args.ids and len(args.subject or []) == 1
        outputs = []
        if single or args.per_subject:
            by_subject = defaultdict(list)
            for row in rows:
                by_subject[row["subject_id"]].append(row)
            outputs += [(out_dir / f"{subject_id}_{feature}.csv", subject_rows)
                        for subject_id, subject_rows in by_subject.items()]
        if not single:
            stem = re.sub(r"[^A-Za-z0-9._+-]+", "_", args.group_name or "merged").strip("_")
            outputs.append((out_dir / f"{stem}_{feature}.csv", rows))
        for path, path_rows in outputs:
            if path.name == summary or path.name.endswith("__" + summary):
                sys.exit(f"{path} would overwrite an input file; choose another --output-dir")
            write_csv(path, header, path_rows)

        print("\n".join(report))
        print(column_summary(info))
        for label, count in info["unmatched"].items():
            print(f"Rows without {label} data (cells left blank): {count}")
        if not single:
            print(f"Wrote merged file {outputs[-1][0]} ({len(rows)} rows)")
        if single or args.per_subject:
            count = len(outputs) - (0 if single else 1)
            print(f"Wrote {count} file(s): {out_dir}/<subject_id>_{feature}.csv")
        if problems:
            print(f"{len(problems)} problem(s):")
            print("\n".join(f"  {p}" for p in problems[:20]))
            if len(problems) > 20:
                print(f"  ... {len(problems) - 20} more")
        if missing:
            path = out_dir / f"missing_{feature}.csv"
            write_csv(path, ["id", "group"], [{"id": i, "group": g} for i, g in missing])
            print(f"IDs without {feature} output: {len(missing)} (listed in {path})")

    if unused:
        print(f"\nWarning: --feature-columns matched nothing: {sorted(unused)}")


if __name__ == "__main__":
    main()
