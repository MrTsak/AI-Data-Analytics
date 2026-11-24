# Browser Dashboard & Fixes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the original diabetes dashboard's bugs and ship a static, interactive browser version for Tsakioan.com.

**Architecture:** A shared `analysis/` Python package (data → models → clustering) feeds both the desktop app (`main.py`) and `scripts/export_web.py`, which writes JSON into `web/data/`. `web/` is a no-build static site (vanilla ES modules plus Observable Plot from a CDN) that renders the precomputed charts and runs both models in JS.

**Tech Stack:** Python 3.10+, pandas, scikit-learn, pytest; vanilla JS ES modules, Observable Plot 0.6.16 + d3 7.9.0 (jsdelivr, pinned); Node 22 `node --test`; Playwright (pre-installed) for the smoke test.

**Spec:** `docs/superpowers/specs/2026-11-24-browser-dashboard-design.md`

## Global Constraints

- Static files only; every path in `web/` is relative (works from `tsakioan.com/projects/diabetes/`).
- External scripts only from pinned jsdelivr versions: `d3@7.9.0`, `@observablehq/plot@0.6.16`.
- Drop exact duplicate rows before splitting; exclude gender "Other" from modelling.
- Smoking history is one-hot encoded; there is no ordinal smoking feature.
- The best model is chosen by ROC-AUC.
- Risk bands: Low < 0.40 ≤ Moderate < 0.70 ≤ High, labelled "model probability".
- Disclaimer "Educational project — not a medical device or diagnosis." appears in the hero and next to the predictor.
- Split: stratified 80/20, `random_state=42`. RF: 150 trees, `max_depth=7`, `min_samples_split=8`, balanced. LR: `StandardScaler` + `LogisticRegression(C=0.1, class_weight='balanced', max_iter=1000)`.
- JS and Python probabilities agree within 1e-6.
- `requirements.txt` is UTF-8.

## Review Focus

1. Slider set to an extreme value (BMI 95, age 0.08) → the predictor still returns a finite probability in [0,1] (Task 4 parity cases include min/max inputs).
2. A feature value exactly equal to an RF split threshold → JS takes the same branch as sklearn (`<=` goes left) (Task 4 parity cases include threshold-equal inputs).
3. Typed out-of-range or empty number input → clamped to the data min/max, never NaN (Task 6 test `clampInput`).
4. Page served from a sub-path → all data fetches resolve (Task 7 smoke test serves `web/` under `/projects/diabetes/`).
5. Desktop app on a dataset with no "Other" gender, or with missing numeric values → it still loads (Task 2 test `test_load_handles_missing_values`).

## File Structure

```
analysis/__init__.py
analysis/data.py          load_dataset(path) -> Dataset
analysis/models.py        train_models(ds) -> ModelResults
analysis/clustering.py    cluster_patients(df) -> ClusterResult
scripts/export_web.py     main(out_dir) — writes web/data/*.json
web/index.html, web/styles.css
web/js/models.js          predictLR, predictRF, riskBand, clampInput (pure; Node-testable)
web/js/charts.js          chart builders (Plot)
web/js/app.js             load JSON, render sections, wire predictor + theme
web/data/                 generated, committed
tests/test_data.py, tests/test_models.py, tests/test_clustering.py, tests/test_export.py
tests/js/models.test.mjs  parity + clamp tests
tests/smoke_web.py        Playwright smoke test
main.py                   refactored desktop app
requirements.txt, requirements-dev.txt, README.md
```

---

### Task 1: Data module + project setup

**Files:** Create `analysis/__init__.py`, `analysis/data.py`, `tests/test_data.py`, `requirements-dev.txt`; rewrite `requirements.txt` (UTF-8).

**Interfaces — Produces:**
- `FEATURES: list[str]` = `['gender_male','age','hypertension','heart_disease','bmi','HbA1c_level','blood_glucose_level','smoking_never','smoking_former','smoking_current','smoking_not_current','smoking_ever']` ("No Info" is the baseline, all zeros).
- `SMOKING_LEVELS: dict[str,str]` raw value → column (excluding `No Info`).
- `@dataclass Dataset: raw: DataFrame` (deduped, no "Other", original columns), `X: DataFrame` (FEATURES), `y: Series`, `n_original: int`, `n_duplicates: int`, `n_other_gender: int`.
- `load_dataset(path='data/sample.csv') -> Dataset`; `encode_patient(d: dict) -> DataFrame` (one row, FEATURES order; input keys: gender 'Female'/'Male', smoking raw value, plus numeric keys).

- [ ] Step 1: Write the tests: `test_drops_duplicates` (n_duplicates == 3854; len(raw) == 100000-3854-n_other_gender after dedupe), `test_excludes_other_gender`, `test_features_one_hot` (X columns == FEATURES; each row's smoking columns sum ≤ 1), `test_encode_patient_matches_X` (encoding a raw row equals that row of X), `test_load_handles_missing_values` (a tmp CSV with a blank bmi loads with no NaNs in X).
- [ ] Step 2: `pip install -r requirements-dev.txt`, then run `pytest tests/test_data.py -v` → FAIL (ImportError).
- [ ] Step 3: Implement. Dedupe first, then drop Other, then fill numeric NaN with the column mean.
- [ ] Step 4: `pytest tests/test_data.py -v` → PASS.
- [ ] Step 5: Commit `feat: shared data loading with dedupe and one-hot smoking`.

### Task 2: Models + clustering modules

**Files:** Create `analysis/models.py`, `analysis/clustering.py`, `tests/test_models.py`, `tests/test_clustering.py`.

**Interfaces — Consumes:** `Dataset`, `FEATURES`. **Produces:**
- `@dataclass ModelResults: rf, lr (fitted Pipeline), X_test, y_test, metrics: dict[name -> {accuracy, precision, recall, f1, auc, confusion: list[list[int]], roc: {fpr, tpr} (≤101 pts)}], best: str`. Names are `"Random Forest"` and `"Logistic Regression"`.
- `train_models(ds: Dataset) -> ModelResults`.
- `@dataclass ClusterResult: labels: ndarray, centers: DataFrame (cols age,bmi,HbA1c_level; index 0..2), sizes: list[int], diabetes_rate: list[float], descriptions: list[str]`.
- `cluster_patients(raw: DataFrame, seed=42) -> ClusterResult`. Order the clusters by mean HbA1c ascending, so cluster 0 is the lowest. The descriptions come from the centres, e.g. `"Avg age 38, BMI 26.1, HbA1c 5.2% — diabetes rate 3.1%"`, prefixed with a lowest/middle/highest-risk label ranked by diabetes rate.

- [ ] Step 1: Write the tests: `test_metrics_sane` (both AUC > 0.9; all metrics in [0,1]; confusion sums to len(y_test)), `test_best_is_max_auc`, `test_roc_downsampled` (≤101 points, starts at 0, ends at 1), `test_cluster_order_stable` (seeds 0 and 42 give the same centres, within 0.05 after ordering, and HbA1c is ascending), `test_descriptions_match_data` (the rate in the text matches diabetes_rate[i] formatted to 1 dp).
- [ ] Step 2: Run them → FAIL.
- [ ] Step 3: Implement. Downsample the ROC by interpolating tpr at `np.linspace(0,1,101)`. KMeans uses `n_init=10` on standardized features.
- [ ] Step 4: `pytest tests -v` → PASS.
- [ ] Step 5: Commit `feat: model training (AUC selection) and stable clustering`.

### Task 3: Desktop app refactor + bug fixes

**Files:** Modify `main.py`.

**Consumes:** `load_dataset`, `encode_patient`, `train_models`, `cluster_patients`.

- [ ] Step 1: Replace `load_data`/`train_models`/`perform_clustering` with calls to `analysis`. Correlate `raw` numeric columns + diabetes only. Use the generated cluster descriptions. Show AUC in the metrics table, and use "BEST MODEL (by AUC)".
- [ ] Step 2: Live prediction: add a gender combo (Female/Male) and a smoking combo with the raw values; build the input via `encode_patient`. Sliders use `command=`. Age goes from 0 to 80 in 80 steps. The scatter uses a 3000-row sample (`random_state=42`).
- [ ] Step 3: Replace bare `except:` with `except tk.TclError:` or a specific exception. A load failure shows a `tkinter.messagebox.showerror` and exits from `main()` only.
- [ ] Step 4: Verify with `xvfb-run -a timeout 20 python main.py` if available (expect no traceback; timeout exit 124). Otherwise run `python -c "import ast;ast.parse(open('main.py').read())"` and `python -c "import main"` with `DISPLAY` unset to confirm the imports.
- [ ] Step 5: Commit `fix: desktop app uses shared pipeline; gender input, slider labels, cluster text`.

### Task 4: Web export + JS model runtime with parity test

**Files:** Create `scripts/export_web.py`, `web/js/models.js`, `tests/test_export.py`, `tests/js/models.test.mjs`; generate `web/data/`.

**Produces (JSON in `web/data/`):**
- `summary.json`: `{n_original, n_duplicates, n_other_gender, n_rows, positive_rate, best, best_auc}`
- `explore.json`: `{histograms: {feat: {edges, neg, pos}} for age,bmi,HbA1c_level,blood_glucose_level (30 bins)}, categories: {gender|smoking_history|hypertension|heart_disease: {labels, neg, pos}}, correlation: {labels, matrix}}`
- `clusters.json`: `{centers: [{age,bmi,HbA1c_level}], sizes, diabetes_rate, descriptions, sample: [{HbA1c_level,bmi,age,cluster}] (3000)}`
- `models.json`: `{metrics (as ModelResults.metrics), best, importances: {feature: rf_importance}, lr_coefs: {feature: coef}}`
- `model_lr.json`: `{features, mean, scale, coef, intercept}`
- `model_rf.json`: `{features, trees: [{f: int[], t: float[], l: int[], r: int[], v: float[]}]}`. Leaves have `f = -1`. `v` is the class-1 probability (the fraction from `tree_.value` normalised per node).
- `form.json`: `{numeric: {feat: {min, max, p01, p99, mean, step}}, smoking: [{value,label}]}`
- `parity.json`: `[{input: {FEATURES...}, lr: p, rf: p}]` (200 cases: 190 rows sampled from X_test, 5 all-min/all-max/mixed extremes, 5 with a feature set exactly to an RF root threshold)

**JS (`web/js/models.js`, pure ES module):** `predictLR(model, x: number[]) -> number`, `predictRF(model, x: number[]) -> number` (left if `x[f] <= t`), `vectorize(features: string[], input: object) -> number[]`, `riskBand(p) -> 'Low'|'Moderate'|'High'`, `clampInput(value: string|number, {min,max,mean}) -> number` (empty/NaN → mean).

- [ ] Step 1: Write the tests. `tests/test_export.py::test_export_schema` runs `main(tmp_path)`, asserts all 8 files exist with the keys above, `model_rf.json` < 2 MB, and the RF predict from the exported arrays (Python walker) equals `rf.predict_proba` within 1e-9. `tests/js/models.test.mjs`: parity for every case in `web/data/parity.json` within 1e-6 for both models; `riskBand(0.39)=='Low'`, `riskBand(0.4)=='Moderate'`, `riskBand(0.7)=='High'`; `clampInput('', r)==r.mean`, `clampInput(999, r)==r.max`.
- [ ] Step 2: Run → FAIL.
- [ ] Step 3: Implement the export, then run `python scripts/export_web.py` (default out `web/data`), then implement `models.js`.
- [ ] Step 4: `pytest tests -v && node --test tests/js/` → PASS. If RF > 2 MB, drop to 100 trees provided ΔAUC < 0.005; otherwise round thresholds to 6 significant figures.
- [ ] Step 5: Commit `feat: export pipeline results and models to JSON with JS parity`.

### Task 5: Web page shell, theme and static sections

**Files:** Create `web/index.html`, `web/styles.css`, `web/js/app.js`, `web/js/charts.js`.

**Consumes:** the `web/data/*.json` schema from Task 4.

- [ ] Step 1: Build `index.html` with the section order from the spec (hero, Explore, Clusters, Models, Try it, Findings) and a sticky nav with anchors. Use CSS colour tokens on `:root`, a dark-mode override via `prefers-color-scheme` plus `[data-theme]`, and a toggle saved to localStorage inside try/catch. Phone layout: single column at ≤ 720px, 16px gutters, no horizontal scroll.
- [ ] Step 2: In `charts.js`, add `histogram(data, feat)`, `categoryBars(data, key)`, `correlationHeatmap(corr)`, `clusterScatter(sample)`, `rocChart(metrics)`, `confusion(cm, title)`, `importanceBars(obj)`. Each returns a DOM node from `Plot.plot`, with colours taken from the CSS tokens. Re-render on theme change and on resize (debounced).
- [ ] Step 3: In `app.js`, fetch the JSON via `new URL('data/x.json', import.meta.url)`-relative paths, render the hero stats, the Explore feature picker and the four hypotheses with their source links (copied from `main.py`), cluster cards, the metrics table with the best value per row highlighted, and Findings (computed from data, with a GitHub link to `https://github.com/MrTsak/AI-Data-Analytics`).
- [ ] Step 4: Serve with `python -m http.server -d web 8000`. Screenshot desktop (1280) and phone (390) widths with Playwright in light and dark, and check visually that no section is empty and nothing overflows.
- [ ] Step 5: Commit `feat: static dashboard page with explore, clusters and model sections`.

### Task 6: Live predictor

**Files:** Modify `web/index.html`, `web/js/app.js`, `web/styles.css`.

**Consumes:** `predictLR`, `predictRF`, `vectorize`, `riskBand`, `clampInput`; `form.json`.

- [ ] Step 1: Build the form: gender (Female/Male segmented control), range + number input pairs for age, bmi, HbA1c_level and blood_glucose_level (range spans p01–p99; the number input allows min–max via `clampInput` on change), checkboxes for hypertension and heart_disease, and a smoking select. Defaults are the data means; inputs have `<label>`s.
- [ ] Step 2: On every `input` event, compute both probabilities and update two result cards (percentage, band, bar width). Show the disclaimer below them.
- [ ] Step 3: Add a test to `tests/js/models.test.mjs`: `vectorize(features, {gender:'Male', smoking_history:'former', ...})` sets `gender_male=1`, `smoking_former=1`, and the other smoking columns to 0.
- [ ] Step 4: `node --test tests/js/` → PASS. Check manually in Playwright that moving the HbA1c slider from 4 to 8.5 raises both probabilities.
- [ ] Step 5: Commit `feat: live dual-model predictor`.

### Task 7: Smoke test, README, final verification

**Files:** Create `tests/smoke_web.py`; modify `README.md`; add `docs/screenshots/*.png`.

- [ ] Step 1: Write `tests/smoke_web.py` (pytest, skipped if playwright is missing). It starts an `http.server` with `web/` copied under `tmp/projects/diabetes/`, opens `/projects/diabetes/`, asserts each section has an `svg` or a table, records no console errors, sets the HbA1c slider to max, and asserts the RF percentage text changed.
- [ ] Step 2: `pytest tests -v && node --test tests/js/` → all PASS.
- [ ] Step 3: Update the README with a live-demo link placeholder (`https://tsakioan.com/projects/diabetes/`), the project structure, how to run the desktop app, how to regenerate `web/data`, deploying by copying `web/` to the site, the test commands, the data-cleaning notes (duplicates, "Other"), the disclaimer, and screenshots from Task 5.
- [ ] Step 4: Commit `docs: README for web version; smoke test`, then push to `ccr-b127e3e2-08fblo`.
