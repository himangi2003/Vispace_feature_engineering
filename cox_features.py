#!/usr/bin/env python3
r"""Cox proportional-hazards analysis of engineered features (one row per subject).

Input: the combined, per-subject table, e.g. from

    python merge_engineered_features.py ... --output all_groups_engineered_features.csv
    python aggregate_subjects.py all_groups_engineered_features.csv --method median

Usage:
    # univariable HR per feature
    python cox_features.py all_groups_engineered_features_median.csv --output cox_results.csv

    # adjusted for age and stage, plus all features in one model
    python cox_features.py features_median.csv --adjust AGE stage --multivariable \
        --output cox_adjusted.csv

    # stage-wise: separate models within stage I-II and III-IV (and all patients)
    python cox_features.py features_median.csv --by stage --stage-groups I,II III,IV \
        --output cox_by_stage.csv

    # one baseline hazard per stage instead of a stage term
    python cox_features.py features_median.csv --strata stage --output cox_strat.csv

Survival: --time (default OS_MONTHS) and --event (default OS_STATUS; values
like "1:DECEASED"/"0:LIVING", 1/0, dead/alive or deceased/living are read).
--limit censors follow-up at that many months (0 = no limit, default).

Features: every numeric column that is not an ID, clinical or survival column
or a *_missing imputation flag (or --features, wildcards allowed). Continuous features are re-standardised
in each analysed set, so HRs are per 1 SD; 0/1 features give the HR for 1 vs 0.
HR > 1 means higher hazard (worse survival) with higher values.

Covariates (--adjust): numeric columns enter as they are (e.g. AGE per year);
text columns as 0/1 indicators against their most common value. The stage
column (--stage-column, default stage; "Stage IIB" etc. is read as II) is
coded by --stage-coding: categorical (I reference), ordinal (I=1..IV=4) or
binary (III-IV vs I-II).

--strata COL fits a stratified Cox model (separate baseline per level).
--by COL repeats everything within each level of COL, after an "All" cohort;
with --by on the stage column, --stage-groups merges stages (e.g. I,II III,IV).
--multivariable adds one model with all features together; keep about 10
events per term (the output warns below that).

Ties: Efron. Missing values: complete cases per model (n is reported).
Output CSV, one row per term:
    cohort, model, feature, term, n, events, hr, ci_low, ci_high, p, q,
    concordance, converged, note
q = Benjamini-Hochberg across features within each cohort and model.
Standard library only.
"""
import argparse
import csv
import fnmatch
import math
import re
import statistics as st
import sys
from collections import Counter, defaultdict
from pathlib import Path

IDS = {"group", "subject_id", "sample_id", "wsi_name", "n_wsi"}
MISSING = {"", "na", "nan", "none", "null"}
PATIENT_PATTERNS = ["age", "diagnosis age", "os_*", "os status", "dfs_*", "pfs_*", "dss_*", "*months*",
                    "*status*", "*days*", "*years*", "*stage*", "*event*", "sex", "gender",
                    "vital_status", "*survival*", "tcga_case_id", "patient_id", "case_id"]
ROMAN = {"I": 1, "II": 2, "III": 3, "IV": 4}


# ----------------------------------------------------------------------------- input
def is_missing(v):
    return v is None or str(v).strip().lower() in MISSING


def number(v):
    if is_missing(v):
        return None
    try:
        x = float(v)
    except ValueError:
        return None
    return x if math.isfinite(x) else None


def matches(column, patterns):
    return any(fnmatch.fnmatchcase(column.lower(), p.lower()) for p in patterns)


def parse_event(v):
    if is_missing(v):
        return None
    text = str(v).strip().lower()
    m = re.match(r"^([01])(\.0+)?(\b|:)", text)
    if m:
        return int(m.group(1))
    if text in ("dead", "deceased", "death", "yes", "true"):
        return 1
    if text in ("alive", "living", "censored", "no", "false"):
        return 0
    return None


def parse_stage(v):
    """'Stage IIB', 'IIIA', '2', 'IV' -> 1..4 (None if unreadable)."""
    if is_missing(v):
        return None
    text = str(v).strip().upper().replace("STAGE", "").strip()
    if text[:1].isdigit():
        return int(text[0]) if text[0] in "1234" else None
    m = re.match(r"^(IV|III|II|I)", text)
    return ROMAN[m.group(1)] if m else None


# ----------------------------------------------------------------------------- linear algebra
def solve(a, b):
    """Solve a x = b (Gauss-Jordan with partial pivoting); None if singular."""
    n = len(a)
    m = [row[:] + [b[i]] for i, row in enumerate(a)]
    for c in range(n):
        piv = max(range(c, n), key=lambda r: abs(m[r][c]))
        if abs(m[piv][c]) < 1e-12:
            return None
        m[c], m[piv] = m[piv], m[c]
        for r in range(n):
            if r != c and m[r][c]:
                f = m[r][c] / m[c][c]
                m[r] = [x - f * y for x, y in zip(m[r], m[c])]
    return [m[i][n] / m[i][i] for i in range(n)]


def inverse(a):
    n = len(a)
    cols = [solve(a, [float(i == j) for i in range(n)]) for j in range(n)]
    if any(c is None for c in cols):
        return None
    return [[cols[j][i] for j in range(n)] for i in range(n)]


# ----------------------------------------------------------------------------- Cox model
def cox_terms(times, events, X, strata, beta):
    """Efron partial log-likelihood, gradient and Hessian."""
    p = len(beta)
    loglik, grad = 0.0, [0.0] * p
    hess = [[0.0] * p for _ in range(p)]
    groups = defaultdict(list)
    for i, s in enumerate(strata):
        groups[s].append(i)
    for idx in groups.values():
        idx.sort(key=lambda i: -times[i])
        s0, s1 = 0.0, [0.0] * p
        s2 = [[0.0] * p for _ in range(p)]
        k = 0
        while k < len(idx):
            t = times[idx[k]]
            same = []
            while k < len(idx) and times[idx[k]] == t:
                same.append(idx[k])
                k += 1
            deaths = [i for i in same if events[i]]
            for i in same:  # risk set = time >= t
                w = math.exp(sum(b * x for b, x in zip(beta, X[i])))
                s0 += w
                for a in range(p):
                    s1[a] += w * X[i][a]
                    for c in range(a, p):
                        s2[a][c] += w * X[i][a] * X[i][c]
            d = len(deaths)
            if not d:
                continue
            d0, d1 = 0.0, [0.0] * p
            d2 = [[0.0] * p for _ in range(p)]
            for i in deaths:
                xb = sum(b * x for b, x in zip(beta, X[i]))
                w = math.exp(xb)
                loglik += xb
                d0 += w
                for a in range(p):
                    grad[a] += X[i][a]
                    d1[a] += w * X[i][a]
                    for c in range(a, p):
                        d2[a][c] += w * X[i][a] * X[i][c]
            for l in range(d):
                f = l / d
                den = s0 - f * d0
                loglik -= math.log(den)
                m1 = [(s1[a] - f * d1[a]) / den for a in range(p)]
                for a in range(p):
                    grad[a] -= m1[a]
                    for c in range(a, p):
                        m2 = (s2[a][c] - f * d2[a][c]) / den
                        hess[a][c] -= m2 - m1[a] * m1[c]
    for a in range(p):
        for c in range(a):
            hess[a][c] = hess[c][a]
    return loglik, grad, hess


def fit_cox(times, events, X, strata, max_iter=60):
    p = len(X[0])
    beta = [0.0] * p
    ll0, grad, hess = cox_terms(times, events, X, strata, beta)
    loglik, converged = ll0, False
    for _ in range(max_iter):
        step = solve([[-h for h in row] for row in hess], grad)
        if step is None:
            break
        scale, new = 1.0, None
        while scale > 1e-4:
            cand = [b + scale * s for b, s in zip(beta, step)]
            ll_new, g_new, h_new = cox_terms(times, events, X, strata, cand)
            if ll_new >= loglik - 1e-12:
                new = (cand, ll_new, g_new, h_new)
                break
            scale /= 2
        if new is None:
            break
        change = new[1] - loglik
        beta, loglik, grad, hess = new
        if abs(change) < 1e-10 and max(abs(s) for s in step) * scale < 1e-8:
            converged = True
            break
        if max(abs(b) for b in beta) > 30:  # separation: coefficient running away
            break
    cov = inverse([[-h for h in row] for row in hess])
    se = [math.sqrt(cov[i][i]) if cov and cov[i][i] > 0 else float("nan") for i in range(p)]
    return dict(beta=beta, se=se, loglik=loglik, loglik0=ll0, converged=converged)


def concordance(times, events, risk):
    """Harrell's C (higher risk should mean shorter survival)."""
    num = den = 0.0
    n = len(times)
    for i in range(n):
        if not events[i]:
            continue
        for j in range(n):
            if times[j] > times[i]:
                den += 1
                num += 1.0 if risk[i] > risk[j] else 0.5 if risk[i] == risk[j] else 0.0
    return num / den if den else float("nan")


def bh(pvalues):
    idx = sorted((p, i) for i, p in enumerate(pvalues) if p is not None and p == p)
    q, prev = [None] * len(pvalues), 1.0
    for rank, (p, i) in reversed(list(enumerate(idx, 1))):
        prev = min(prev, p * len(idx) / rank)
        q[i] = prev
    return q


# ----------------------------------------------------------------------------- design
def covariate_columns(rows, adjust, stage_column, stage_coding):
    """[(term name, function row -> value or None)] for the adjustment covariates."""
    terms = []
    for col in adjust:
        if col == stage_column:
            if stage_coding == "ordinal":
                terms.append((f"{col} (I-IV ordinal)", lambda r, c=col: parse_stage(r[c])))
            elif stage_coding == "binary":
                terms.append((f"{col} III-IV vs I-II", lambda r, c=col: None if parse_stage(r[c]) is None
                              else float(parse_stage(r[c]) >= 3)))
            else:
                levels = sorted({parse_stage(r[col]) for r in rows} - {None})
                for lv in levels[1:]:
                    name = {1: "I", 2: "II", 3: "III", 4: "IV"}[lv]
                    terms.append((f"{col} {name} vs I" if levels[0] == 1 else f"{col} {name}",
                                  lambda r, c=col, v=lv: None if parse_stage(r[c]) is None
                                  else float(parse_stage(r[c]) == v)))
            continue
        values = [r[col] for r in rows if not is_missing(r[col])]
        if values and all(number(v) is not None for v in values):
            terms.append((col, lambda r, c=col: number(r[c])))
        else:
            counts = Counter(v.strip() for v in values)
            if len(counts) < 2:
                continue
            ref = counts.most_common(1)[0][0]
            for lv in sorted(counts):
                if lv != ref:
                    terms.append((f"{col}={lv} vs {ref}", lambda r, c=col, v=lv:
                                  None if is_missing(r[c]) else float(r[c].strip() == v)))
    return terms


def run_model(rows, time_col, event_col, limit, feature_terms, cov_terms, strata_col, binary):
    """Fit one Cox model; returns rows of results (one per feature term) or a note."""
    data = []
    for r in rows:
        t, e = number(r[time_col]), parse_event(r[event_col])
        if t is None or e is None or t < 0:
            continue
        if limit and t > limit:
            t, e = limit, 0
        xs = [f(r) for _, f in feature_terms] + [f(r) for _, f in cov_terms]
        s = r[strata_col].strip() if strata_col else ""
        if any(x is None for x in xs) or (strata_col and is_missing(s)):
            continue
        if strata_col and strata_col == STAGE_COLUMN[0] and parse_stage(s) is not None:
            s = str(parse_stage(s))
        data.append((t, e, xs, s))
    n, events = len(data), sum(d[1] for d in data)
    names = [nm for nm, _ in feature_terms] + [nm for nm, _ in cov_terms]
    if n < 10 or events < 3:
        return [], n, events, f"too few rows/events (n={n}, events={events})"
    cols = list(zip(*[d[2] for d in data]))
    # Re-standardise continuous features within this set; drop constant columns.
    keep, scaled = [], []
    for j, (name, col) in enumerate(zip(names, cols)):
        is_feature = j < len(feature_terms)
        sd = st.pstdev(col)
        if sd == 0:
            continue
        if is_feature and name not in binary:
            m = st.fmean(col)
            col = [(x - m) / sd for x in col]
        keep.append(j)
        scaled.append(list(col))
    if not any(j < len(feature_terms) for j in keep):
        return [], n, events, "feature has no variation in this set"
    X = [list(row) for row in zip(*scaled)]
    times, evs, strata = [d[0] for d in data], [d[1] for d in data], [d[3] for d in data]
    fit = fit_cox(times, evs, X, strata)
    risk = [sum(b * x for b, x in zip(fit["beta"], row)) for row in X]
    c = concordance(times, evs, risk)
    out = []
    for k, j in enumerate(keep):
        b, se = fit["beta"][k], fit["se"][k]
        z = b / se if se and se == se and se > 0 else float("nan")
        p = math.erfc(abs(z) / math.sqrt(2)) if z == z else float("nan")
        out.append(dict(term=names[j], is_feature=j < len(feature_terms), n=n, events=events,
                        hr=math.exp(b), ci_low=math.exp(b - 1.959964 * se) if se == se else float("nan"),
                        ci_high=math.exp(b + 1.959964 * se) if se == se else float("nan"),
                        p=p, concordance=c, converged=fit["converged"],
                        note="" if fit["converged"] else "did not converge (possible separation)"))
    return out, n, events, ""


STAGE_COLUMN = ["stage"]


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", type=Path, help="Per-subject feature CSV")
    parser.add_argument("--output", type=Path, default=Path("cox_results.csv"))
    parser.add_argument("--time", default="OS_MONTHS")
    parser.add_argument("--event", default="OS_STATUS")
    parser.add_argument("--limit", type=float, default=0, help="Censor at this many months (0 = none)")
    parser.add_argument("--features", nargs="+", help="Feature columns (names or wildcards)")
    parser.add_argument("--adjust", nargs="+", default=[], help="Covariates, e.g. AGE stage")
    parser.add_argument("--stage-column", default="stage")
    parser.add_argument("--stage-coding", choices=["categorical", "ordinal", "binary"], default="categorical")
    parser.add_argument("--strata", help="Column for a stratified model (separate baseline per level)")
    parser.add_argument("--by", help="Run separately within each level of this column (plus All)")
    parser.add_argument("--stage-groups", nargs="+", metavar="I,II",
                        help="With --by on the stage column: merge stages into groups, e.g. I,II III,IV")
    parser.add_argument("--multivariable", action="store_true", help="Also fit all features in one model")
    parser.add_argument("--no-all", action="store_true", help="With --by, skip the All cohort")
    args = parser.parse_args()
    STAGE_COLUMN[0] = args.stage_column

    with open(args.input, newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        header = [c.strip() for c in reader.fieldnames or []]
        rows = [{k.strip(): (v or "").strip() for k, v in r.items() if k is not None} for r in reader]
    if not rows:
        sys.exit(f"{args.input}: no data rows")
    for col in [args.time, args.event] + args.adjust + [c for c in (args.strata, args.by) if c]:
        if col not in header:
            sys.exit(f"{args.input}: no column {col!r}")
    key = [c for c in ("group", "subject_id") if c in header]
    dupes = [k for k, n in Counter(tuple(r[c] for c in key) for r in rows).items() if n > 1]
    if dupes:
        sys.exit(f"{args.input}: {len(dupes)} subjects have several rows (e.g. {dupes[0]}); "
                 "aggregate first with aggregate_subjects.py")
    bad_events = sorted({r[args.event] for r in rows if not is_missing(r[args.event]) and parse_event(r[args.event]) is None})
    if bad_events:
        sys.exit(f"{args.event}: cannot read event values {bad_events[:5]}")

    exclude = set(args.adjust) | {args.time, args.event, args.strata, args.by} | IDS
    numeric = [c for c in header if c not in exclude
               and any(not is_missing(r[c]) for r in rows)
               and all(is_missing(r[c]) or number(r[c]) is not None for r in rows)]
    if args.features:
        features = [c for c in numeric if matches(c, args.features)]
        if not features:
            sys.exit("--features matched no numeric columns")
    else:
        # *_missing flags only mark imputed values; name them in --features to test them.
        features = [c for c in numeric if not matches(c, PATIENT_PATTERNS) and not c.endswith("_missing")]
    binary = {c for c in features if {r[c] for r in rows if not is_missing(r[c])} <= {"0", "1"}}

    # Cohorts
    cohorts = [] if (args.by and args.no_all) else [("All", rows)]
    if args.by:
        if args.by == args.stage_column:
            if args.stage_groups:
                for spec in args.stage_groups:
                    wanted = {ROMAN.get(s.strip().upper()) for s in spec.split(",")}
                    if None in wanted:
                        sys.exit(f"--stage-groups: cannot read {spec!r} (use roman numerals, e.g. I,II)")
                    label = "stage " + "-".join(s.strip().upper() for s in spec.split(","))
                    cohorts.append((label, [r for r in rows if parse_stage(r[args.by]) in wanted]))
            else:
                for lv in sorted({parse_stage(r[args.by]) for r in rows} - {None}):
                    cohorts.append((f"stage {['I', 'II', 'III', 'IV'][lv - 1]}",
                                    [r for r in rows if parse_stage(r[args.by]) == lv]))
        else:
            for lv in sorted({r[args.by] for r in rows if not is_missing(r[args.by])}):
                cohorts.append((f"{args.by}={lv}", [r for r in rows if r[args.by] == lv]))

    results = []
    for cohort, cohort_rows in cohorts:
        cov_terms = covariate_columns(cohort_rows, args.adjust, args.stage_column, args.stage_coding)
        model = ("adjusted (" + ", ".join(args.adjust) + ")" if args.adjust else "univariable") + \
            (f", strata {args.strata}" if args.strata else "")
        batch = []
        for f in features:
            out, n, ev, note = run_model(cohort_rows, args.time, args.event, args.limit,
                                         [(f, lambda r, c=f: number(r[c]))], cov_terms, args.strata, binary)
            if note:
                batch.append(dict(cohort=cohort, model=model, feature=f, term=f, n=n, events=ev, note=note))
            for o in out:
                batch.append(dict(cohort=cohort, model=model, feature=f, **o))
        feature_rows = [b for b in batch if b.get("is_feature")]
        for b, q in zip(feature_rows, bh([b["p"] for b in feature_rows])):
            b["q"] = q
        results += batch
        if args.multivariable:
            out, n, ev, note = run_model(cohort_rows, args.time, args.event, args.limit,
                                         [(f, lambda r, c=f: number(r[c])) for f in features],
                                         cov_terms, args.strata, binary)
            mv = "multivariable" + (f" + {', '.join(args.adjust)}" if args.adjust else "") + \
                (f", strata {args.strata}" if args.strata else "")
            terms = len(out)
            warn = f"{ev / terms:.1f} events per term (< 10)" if terms and ev / terms < 10 else ""
            if note:
                results.append(dict(cohort=cohort, model=mv, feature="(all)", term="(all)", n=n, events=ev, note=note))
            fr = [o for o in out if o["is_feature"]]
            for o, q in zip(fr, bh([o["p"] for o in fr])):
                o["q"] = q
            for o in out:
                results.append(dict(cohort=cohort, model=mv, feature=o["term"] if o["is_feature"] else "(covariate)",
                                    **{**o, "note": "; ".join(x for x in (o["note"], warn) if x)}))

    cols = ["cohort", "model", "feature", "term", "n", "events", "hr", "ci_low", "ci_high", "p", "q",
            "concordance", "converged", "note"]
    with open(args.output, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(cols)
        for r in results:
            writer.writerow([("" if r.get(c) is None else
                              (f"{r[c]:.6g}" if isinstance(r.get(c), float) else r.get(c, ""))) for c in cols])

    print(f"{args.input.name}: {len(rows)} subjects, {len(features)} features "
          f"({len(binary)} 0/1); time {args.time}, event {args.event}"
          + (f", censored at {args.limit:g}" if args.limit else ""))
    for cohort, cohort_rows in cohorts:
        sub = [r for r in results if r["cohort"] == cohort and r.get("is_feature") and "multivariable" not in r["model"]]
        if not sub:
            print(f"\n[{cohort}] no model could be fitted")
            continue
        print(f"\n[{cohort}] {sub[0]['model']}: n={sub[0]['n']}, events={sub[0]['events']}  "
              "(HR per SD; 0/1 features: 1 vs 0)")
        for r in sorted(sub, key=lambda r: (r["p"] != r["p"], r["p"]))[:12]:
            q = r.get("q")
            flag = " *" if q is not None and q < 0.05 else ""
            print(f"  {r['feature']:36} HR {r['hr']:6.2f} ({r['ci_low']:.2f}-{r['ci_high']:.2f})  "
                  f"p {r['p']:.3g}  q {'-' if q is None else f'{q:.3g}'}{flag}")
    if args.multivariable:
        print("\nMultivariable model rows are in the output (model = multivariable...).")
    print(f"\nWrote {args.output} ({len(results)} rows). * = q < 0.05")


if __name__ == "__main__":
    main()
