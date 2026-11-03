# Whole-slide spatial feature pipeline

These scripts go from the per-WSI ViSpace outputs to survival analysis and figures:

1. combine the per-WSI outputs for a list of subjects, adding clinical and survival columns;
2. stack the groups;
3. aggregate to one row per subject;
4. look at the distributions;
5. engineer the features;
6. summarise the cohort;
7. merge the feature sets;
8. run Cox models;
9. draw forest plots;
10. draw Kaplan–Meier plots;
11. export PNGs.

All the scripts use the **Python standard library only**: no pandas, numpy, matplotlib or lifelines. They need Python 3.9 or later. Every script supports `--help`, which lists every option with its default.

## Files that must sit in the same folder

Some scripts import from others:

| Script | Imports from |
|---|---|
| `tumor_morphology.py` | `combine_wsi_outputs.py` |
| `km_cox_significant.py` | `km_plot_svg.py`, `plot_cox_forest_svg.py`, `svg_to_png.py` |
| `km_plot_svg.py`, `plot_cox_forest_svg.py` (`--png` only) | `svg_to_png.py` |

All the other scripts stand alone.

## Pipeline at a glance

```
<root>/<wsi>/spatial_feature_results/.../*.csv
        │  combine_wsi_outputs.py   (one run per group, e.g. ILC / IDC)
        ▼
<group>_<feature>.csv  (tumor_core, immune_proximity, tils_tsr, necrosis_feature)
        │  concat_group_outputs.py
        ▼
all_groups_<feature>.csv  ──► plot_feature_distributions.py  (HTML)
        │  aggregate_subjects.py   (one row per subject)
        ▼
all_groups_<feature>_median.csv
        │  <feature>_feature_engineering.py   (4 scripts)
        ▼
all_groups_<feature>_engineered.csv ──► feature_cohort_summary.py  (reference table, PCA, HTML)
        │  merge_engineered_features.py
        ▼
all_groups_engineered_features.csv
        │  cox_features.py   (overall, adjusted, by stage)
        ▼
cox_*.csv ──► plot_cox_forest_svg.py   (forest plots)
          ──► km_plot_svg.py           (KM plots for chosen features)
          ──► km_cox_significant.py    (KM grids for features with Cox p ≤ 0.05)
                     │  svg_to_png.py  (every figure script has --png)
                     ▼
                  *.svg + *.png
```

You can stop after any step. For example, KM plots work directly on an aggregated file, without feature engineering.

---

## 1. `combine_wsi_outputs.py`: per-WSI outputs into one file per group

This step finds `<root>/<wsi_name>/spatial_feature_results/.../<summary>.csv` for each WSI. It then writes one row per WSI per feature set, with the chosen clinical and survival columns added.

The feature sets are `tumor_core`, `immune_proximity`, `tils_tsr` and `necrosis_feature`.

**Subjects**

- One subject gives `<subject_id>_<feature>.csv`.
- A list of subjects (`--subject ids.txt` or `--ids ...`) together with `--group-name` gives one merged `<group-name>_<feature>.csv`.
- The search only looks inside the folders of the subjects you list.

**Clinical and survival columns**

- `--clinical master_clinical.csv` adds clinical columns.
  - `--clinical-columns` chooses which ones. The default is all of them; `--clinical-columns` with no names adds none.
- `--survival` and `--survival-columns` work the same way for the survival file.
  - The defaults are `OS_MONTHS` and `OS_STATUS`. The aliases "Overall Survival (Months)" and "Overall Survival Status" are accepted.

**How WSI names are matched to clinical rows**

1. The exact WSI name.
2. The normalised name, with the folder, file extension and letter case removed.
3. The sample ID.
4. The patient ID.

When several clinical rows match, only the values they share are copied. The script prints a "Row matching:" summary so you can see how many rows matched at each level.

There is no `source_file` column. `merge_subject_clinical.py` has been folded into this script.

**Other options**

| Option | Purpose |
|---|---|
| `--root` | One or more root folders |
| `--features` | Feature sets to combine |
| `--feature-columns` | Feature columns to keep |
| `--keep-prefixes` | Column prefixes to keep |
| `--list-columns` | Show the available columns and exit |
| `--output-dir` | Output folder |
| `--per-subject` | Also write one file per subject |
| `--wsi-column` | Name of the WSI column in the clinical file |
| `--group-names` | Names for several groups in one run |

```bash
python combine_wsi_outputs.py --root /data/vispace_out --subject ilc_ids.txt --group-name ILC \
    --clinical master_clinical.csv --clinical-columns age stage ER_status --output-dir combined
python combine_wsi_outputs.py --root /data/vispace_out --subject idc_ids.txt --group-name IDC \
    --clinical master_clinical.csv --clinical-columns age stage ER_status --output-dir combined
```

### `tumor_morphology.py` (companion script)

This script produces the 15 tumor-morphology features, with the same clinical and survival handling. It uses `combine_wsi_outputs.py` for that.

- `--aggregate mean|median` gives one row per subject.
- A survival fix fills `OS_event`, `OS_days` and `OS_months` from the follow-up and death days.
- `--no-survival-fix` turns the survival fix off.

---

## 2. `concat_group_outputs.py`: stack the groups

This script combines `ILC_tumor_core.csv`, `IDC_tumor_core.csv` and so on into `all_groups_tumor_core.csv`. The `group` column holds the group name.

It skips per-subject TCGA files, `missing_*` files and files that already carry the output prefix.

| Option | Purpose |
|---|---|
| `--input-dir` (required) | One or more input folders |
| `--output-dir` (required) | Output folder |
| `--features` | Feature sets to combine (default: all found) |
| `--groups` | Only these groups (default: all found) |
| `--prefix` | Output prefix (default `all_groups`) |

```bash
python concat_group_outputs.py --input-dir combined --output-dir combined
```

## 3. `aggregate_subjects.py`: one row per subject

This script collapses the WSIs of each subject into one row per (group, subject_id).

**Numeric features**

- `--method mean|median` sets the default (median).
- `--method-for FEATURE=METHOD` overrides it for single features. Wildcards are allowed, e.g. `"log_*area_mm2=mean"`.

**0/1 features** (one-hot categories and `*_missing` flags)

- `--binary mode` takes the most frequent value (the default).
- `--binary any` gives 1 if any WSI has 1.
- `--binary mean` gives the share of WSIs with 1.

**Other columns**

- Text features take their most frequent value.
- Clinical and survival columns are copied. The script warns if they differ between a subject's WSIs.
- `wsi_name` and `sample_id` are joined with `;`.
- `n_wsi` is added.

**Options**

| Option | Purpose |
|---|---|
| `--features` | Columns to aggregate (default: every non-clinical column) |
| `--patient-columns` | Columns to copy unchanged |
| `--output-dir` | Folder for `<input>_<method>.csv` |
| `--output` | Output file name (one input only) |
| `--split-groups` | Write one file per group |

```bash
python aggregate_subjects.py combined/all_groups_*.csv --method median --output-dir by_subject
```

## 4. `plot_feature_distributions.py`: interactive distribution pages

This script writes one self-contained HTML page per CSV. Each page has:

- histograms per group (up to 3 groups), with a Raw/log1p toggle;
- the share of each category for text features;
- a Spearman correlation heatmap;
- a table view.

Options: `inputs`, `--features`, `--output-dir`.

```bash
python plot_feature_distributions.py by_subject/all_groups_*_median.csv --output-dir distributions
```

## 5. Feature engineering (four scripts)

There is one script per feature set: `tumor_core_feature_engineering.py`, `tils_tsr_feature_engineering.py`, `immune_proximity_feature_engineering.py` and `necrosis_feature_engineering.py`.

Each script:

1. transforms skewed features (log1p, log, log(x + ε), sqrt, logit; ε = half the smallest positive value);
2. drops redundant duplicates;
3. caps outliers at the 1st and 99th percentiles (`--winsorize`);
4. imputes missing values with the median and adds a `<feature>_missing` flag (`--impute`);
5. converts to z-scores (skip with `--no-standardize`).

**Options shared by all four**

| Option | Purpose |
|---|---|
| `--output` | Output file (default `<input>_engineered.csv`) |
| `--fit-group ILC` | Compute medians, caps and z-score parameters on one group only, then apply them to everyone |
| `--save-params params.json` | Save the fitted parameters |
| `--params params.json` | Apply saved parameters to a new cohort without refitting |
| `--rename` | Rename columns (not in `tumor_core`) |

Text categories with at least 10 rows are one-hot encoded. The most common category is the reference.

**What each script produces**

| Script | Engineered features | Extra options |
|---|---|---|
| tumor_core | `log_tumor_area_mm2`, `tumor_area_fraction`, `log_tumor_edge_per_area_per_mm`, `tumor_largest_island_share`, `sqrt_tumor_island_roundness`, `tumor_island_solidity`, `log_elongation_minus1`, `tumor_edge_irregularity` (+ `_missing`), `log_tumor_island_spacing_um`, `tumor_island_gap_um` | `--fragmentation edge\|islands\|score`, `--keep-dropped` |
| tils_tsr | `log_tsr_ratio`, `sqrt_til_stromal_pct`, `log_til_intratumoral_pct` | `--keep-between`, `--tsr-flag` |
| immune_proximity | `log_til_total_area_mm2`, `logit_til_share_inside_tumor`, `logit_til_share_touching_tumor`, `til_within_50um_pct`, `sqrt_til_beyond_100um_pct`, `til_outside_distance_median_um`, one-hot `immune_phenotype_*` | `--keep-200um` |
| necrosis | `log_necrosis_tissue_fraction`, `log_necrosis_area_mm2` (drops `region_count` and `boundary_length_mm`) | `--keep-boundary`, `--category-flag` |

```bash
python tumor_core_feature_engineering.py by_subject/all_groups_tumor_core_median.csv --save-params tumor_core_params.json
python tils_tsr_feature_engineering.py by_subject/all_groups_tils_tsr_median.csv
python immune_proximity_feature_engineering.py by_subject/all_groups_immune_proximity_median.csv
python necrosis_feature_engineering.py by_subject/all_groups_necrosis_feature_median.csv
```

## 6. `feature_cohort_summary.py`: cohort reference table and PCA

This script works on any feature set.

**Inputs**

- `--raw` (required): the raw feature CSV.
- `--engineered` (required): the engineered version.
- `--tissue-file`: the CSV that holds `analyzed_area_mm2` and `tumor_region_count`, if the raw file doesn't have them. The column names can be changed with `--area-column` and `--region-column`.

**Small tissue**

A sample counts as small tissue if:

- its analysed area is below `--min-area` (default 10 mm²), or
- it has fewer than `--min-regions` tumor regions (default 3).

**Outputs** (each prefixed with `<name>_`)

| File | Contents |
|---|---|
| `reference_table.csv` | Median, IQR and range per group |
| `pca_variance.csv`, `pca_loadings.csv`, `pca_scores.csv` | PCA results; components are kept up to `--variance` cumulative variance (default 0.80, at least 2) |
| `pca_model.json` | The PCA model, reusable on a new cohort |
| `tissue_dependence.csv` | How each feature depends on the amount of tissue |
| `cohort_summary.html` | All of the above as one page |

```bash
python feature_cohort_summary.py --raw by_subject/all_groups_tils_tsr_median.csv \
    --engineered by_subject/all_groups_tils_tsr_median_engineered.csv \
    --tissue-file by_subject/all_groups_tumor_core_median.csv --output-dir cohort_summary
```

## 7. `merge_engineered_features.py`: one analysis table

This script joins the engineered files on `group` + `subject_id`. It also uses `wsi_name` when there are several rows per subject.

- Shared columns (clinical, survival) are written once. They must agree between files; otherwise the script stops. `--allow-conflicts` keeps the first file's value instead.
- `--how outer` (the default) keeps every subject and leaves blanks where a file has no row. `--how inner` keeps only subjects found in every file.
- `--prefix-all` prefixes every feature with its file's short name.

```bash
python merge_engineered_features.py by_subject/*_engineered.csv --output all_groups_engineered_features.csv
```

## 8. `cox_features.py`: Cox proportional-hazards models

This script fits Cox models with Efron ties, using Newton–Raphson with step-halving.

**Output columns**

`cohort, model, feature, term, n, events, hr, ci_low, ci_high, p, q, concordance, converged, note`

- `ci_low` and `ci_high` are Wald 95% CIs.
- `q` is the Benjamini–Hochberg adjusted p.
- `concordance` is Harrell's C.

The estimates were checked against simulated data and brute-force likelihoods.

**Survival**

- `--time` (default `OS_MONTHS`).
- `--event` (default `OS_STATUS`). Accepted values are `1:DECEASED` / `0:LIVING`, 1/0, or yes/no.
- `--limit 120` censors follow-up at 120 months (0 means no limit).

**Models**

- Univariable by default, one model per feature. `*_missing` flags are skipped.
- `--features`: features to model.
- `--adjust age stage`: adjust each feature for these columns.
  - `--stage-column` (default `stage`): "Stage IIB" is read as II.
  - `--stage-coding categorical|ordinal|binary`.
- `--strata group`: a separate baseline hazard per level.
- `--multivariable`: also fit all the features in one model.

**Cohorts**

- `--by stage --stage-groups I,II III,IV` adds a cohort for each set of stages, e.g. "stage I-II".
- `--no-all` leaves out the All cohort.

```bash
python cox_features.py all_groups_engineered_features.csv --limit 120 --output cox_univariable.csv
python cox_features.py all_groups_engineered_features.csv --limit 120 --adjust age stage --output cox_adjusted.csv
python cox_features.py all_groups_engineered_features.csv --limit 120 --by stage --stage-groups I,II III,IV --output cox_by_stage.csv
```

## 9. `plot_cox_forest_svg.py`: forest plots

This script writes one SVG per results file and model. The HRs are on a log scale, with cohorts side by side.

- Features get readable labels. Use `--raw-names` to show the column names instead, or `--labels labels.csv` (columns `feature,label`) to supply your own.
- The square is filled when p < `--alpha` (default 0.05).
  - `--significance q` uses the BH q instead.
  - `--show-q` adds a q column.
- `--sort input|hr|p` orders the rows.
- `--png` also writes PNGs; `--png-scale` sets the pixel density (default 2).

```bash
python plot_cox_forest_svg.py cox_univariable.csv cox_adjusted.csv cox_by_stage.csv --output-dir plots --png
```

## 10. `km_plot_svg.py`: Kaplan–Meier plots for chosen features

This script writes one SVG per feature. Each plot has:

- KM curves with a log-rank p-value;
- median survival;
- a number-at-risk table.

**Input:** one row per subject, with time and event columns (the same defaults as `cox_features.py`).

**Options**

| Option | Purpose |
|---|---|
| `--features` (required) | Features to plot |
| `--split` | `median` (the default), `tertile`, or `cut:X` for a fixed threshold |
| `--cut-within all\|cohort` | Cut-points from the whole cohort (the default), or within each panel |
| `--by stage --stage-groups I,II III,IV` | One panel per set of stages |
| `--no-all` | Leave out the All panel |
| `--limit` | Censor follow-up at this many months |
| `--labels FEATURE=LABEL` | Your own feature labels |
| `--swap-colors` | Swap the two colours |
| `--png`, `--png-scale` | Also write PNGs |

```bash
python km_plot_svg.py all_groups_engineered_features.csv \
    --features log_necrosis_tissue_fraction log_necrosis_area_mm2 \
    --by stage --stage-groups I,II III,IV --limit 120 --output-dir plots --png
```

## 11. `km_cox_significant.py`: KM grids for Cox-significant features

This script reads the Cox results and selects the features with p ≤ `--p` (default 0.05). It then draws combined figures with one row per feature:

- `km_significant_all_patients.svg`: all patients, from `--cox`.
- `km_significant_by_stage.svg`: one column per stage group, plus All patients unless `--no-all`, from `--cox-by-stage`.

Every panel shows that cohort's HR, CI and p from Cox, plus the log-rank p.

**Colours** (`--color-by risk`, the default)

- Orange is the higher-risk group and blue is the lower-risk group.
- The direction comes from the Cox HR. HR > 1 means high values carry the higher risk.
- Gray is used when the direction is unknown.

**Options**

| Option | Purpose |
|---|---|
| `--data` (required) | Per-subject features and survival |
| `--model` | Which Cox model to read (default: the first in each file) |
| `--split median\|tertile` | How to split the feature |
| `--all` | Plot every feature, not only the significant ones |
| `--per-feature` | Also write one figure per feature |
| `--png`, `--png-scale` | Also write PNGs |
| `--time`, `--event`, `--limit`, `--stage-column`, `--stage-groups` | As in `cox_features.py` |

```bash
python km_cox_significant.py --data all_groups_engineered_features.csv \
    --cox cox_univariable.csv --cox-by-stage cox_by_stage.csv \
    --stage-groups I,II III,IV --limit 120 --output-dir plots --png
# every feature, one file each:
python km_cox_significant.py --data all_groups_engineered_features.csv --cox cox_univariable.csv \
    --cox-by-stage cox_by_stage.csv --stage-groups I,II III,IV --limit 120 \
    --all --per-feature --output-dir plots/all_features --png
```

## 12. `svg_to_png.py`: SVG to PNG

The figure scripts call this when you pass `--png`. You can also run it on its own, on files or whole folders.

It tries these converters in order and keeps the first one that works:

1. **cairosvg** (`pip install cairosvg`): recommended, the simplest.
2. **Chrome / Chromium / Edge**, run headless. It is found on PATH, in the usual macOS/Windows install folders or in the Playwright cache, or via `CHROME_PATH=/path/to/chrome`.
3. **rsvg-convert**.
4. **Inkscape 1.x**.

| Option | Purpose |
|---|---|
| `--scale` | Pixel density (default 2) |
| `--output-dir` | Output folder (default: next to each SVG) |
| `--backend cairosvg\|chrome\|rsvg\|inkscape` | Use one converter only |

```bash
python svg_to_png.py plots/                       # every .svg in the folder
python svg_to_png.py plots/km_*.svg --scale 3 --output-dir plots_png
```

> **Note on Chrome:** started with a temporary profile (`--user-data-dir`), Chrome can hang. It only finishes reliably when `--no-first-run`, `--disable-component-update` and `--no-default-browser-check` are also passed. If PNG export hangs, install cairosvg and run with `--backend cairosvg`.

---

## Typical full run

```bash
python combine_wsi_outputs.py --root OUT --subject ilc_ids.txt --group-name ILC --clinical master_clinical.csv --output-dir combined
python combine_wsi_outputs.py --root OUT --subject idc_ids.txt --group-name IDC --clinical master_clinical.csv --output-dir combined
python concat_group_outputs.py --input-dir combined --output-dir combined
python aggregate_subjects.py combined/all_groups_*.csv --method median --output-dir by_subject
python plot_feature_distributions.py by_subject/*_median.csv --output-dir distributions
for f in tumor_core tils_tsr immune_proximity necrosis; do
  in=by_subject/all_groups_${f}_median.csv; [ $f = necrosis ] && in=by_subject/all_groups_necrosis_feature_median.csv
  python ${f}_feature_engineering.py "$in"
done
python merge_engineered_features.py by_subject/*_engineered.csv --output features.csv
python cox_features.py features.csv --limit 120 --output cox_univariable.csv
python cox_features.py features.csv --limit 120 --by stage --stage-groups I,II III,IV --output cox_by_stage.csv
python plot_cox_forest_svg.py cox_univariable.csv cox_by_stage.csv --output-dir plots --png
python km_cox_significant.py --data features.csv --cox cox_univariable.csv --cox-by-stage cox_by_stage.csv \
    --stage-groups I,II III,IV --limit 120 --output-dir plots --png
```
