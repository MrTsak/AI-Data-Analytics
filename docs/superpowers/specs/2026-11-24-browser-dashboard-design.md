# Diabetes Prediction Dashboard — Browser Version & Fixes

**Date:** 2026-11-24
**Status:** Draft, awaiting approval

## Intent

The project began two years ago as a Gatehouse Awards submission: a CustomTkinter desktop app that explores a 100k-record diabetes dataset, compares two classifiers, clusters patients, and offers a live risk estimate. The owner wants to upgrade it, fix its problems, and feature it on their portfolio (Tsakioan.com), where **visitors can play with it directly in the browser**.

Success means a visitor can open the page, get something on screen within about a second, explore the data and model results interactively, and move sliders to see a risk estimate update live. On the GitHub side, a reviewer sees a clean, tested Python pipeline behind it.

## Goals

1. A static web dashboard (HTML/CSS/JS, no server) that can be dropped onto Tsakioan.com.
2. A Python pipeline that trains the models and exports everything the page needs.
3. Fixes for the correctness bugs in the original app. The fixed desktop app stays in the repo as the original Gatehouse version.
4. Tests for the pipeline, plus a test that the browser's predictions match Python's.
5. An updated README with screenshots and a link to the live page.

## Non-goals

- Retraining models in the browser, or uploading custom datasets.
- A backend, accounts, or saving visitor inputs.
- New model families beyond the existing Random Forest and Logistic Regression.
- Integrating into the portfolio site's framework. We ship a self-contained folder; the owner places it.

## Anti-goals (a failure even if it works)

- Slow first load (heavy runtimes such as Pyodide or Plotly's full bundle).
- Presenting the tool as medical advice. The "educational only, not a diagnosis" disclaimer must be visible near the prediction.
- Browser predictions that silently differ from the Python models.
- Headline claims that mislead on imbalanced data, such as "97% accurate" with no recall shown.

## Constraints

- Dataset: `data/sample.csv`, 100,000 rows, 9 columns. 8.5% are diabetic. 3,854 rows are exact duplicates. 18 rows have gender "Other".
- Hosting: static files only, with relative paths so the folder works from any sub-path (e.g. `tsakioan.com/projects/diabetes/`).
- External scripts come only from pinned CDN versions, or are vendored.

## Approaches considered

| Approach | Trade-off | Decision |
|---|---|---|
| **A. Static page, models pre-trained in Python and exported to JSON** | Instant load, free hosting, lives on the owner's own domain. Models are frozen at export time, and the UI must be rewritten in JS. | **Chosen** by the owner |
| B. Streamlit on Community Cloud | Least rewriting, all Python. The free tier sleeps, giving a 30s+ wake screen, and it runs on a third-party domain. | Rejected: poor first impression |
| C. Pyodide (Python in the browser) | Real sklearn in the browser, but a 30–50 MB download before anything appears. | Rejected: too slow |

## Bugs to fix (in both the desktop app and the new pipeline)

| # | Problem | Fix |
|---|---|---|
| 1 | The live prediction hardcodes `gender_num = 1` (always male), and the form has no gender input. | Add a gender input; pass the real value. |
| 2 | The cluster descriptions are hardcoded text, but K-means label order is arbitrary, so the text can describe the wrong cluster. | Sort clusters by a stable key (mean HbA1c, then age) and generate descriptions from the actual cluster centres and diabetes rates. |
| 3 | The "best model" is chosen by accuracy on 8.5%-positive data. | Choose by ROC-AUC, and show accuracy, precision, recall, F1 and AUC side by side. |
| 4 | Slider value labels update only on drag (`<B1-Motion>`), not on click. The age slider starts at 18, but `number_of_steps` assumes a 0.08 minimum. | Use the slider `command=` callback; make range and steps consistent. |
| 5 | Smoking history is encoded as an arbitrary ordinal 0–5. | One-hot encode it. |
| 6 | The correlation heatmap includes the `Cluster` column and derived columns. | Correlate only the original features plus the outcome. |
| 7 | 3,854 exact duplicate rows can land in both the train and test sets, which inflates scores. | Drop exact duplicates before splitting. |
| 8 | `requirements.txt` is UTF-16 encoded. | Re-save as UTF-8 and pin sensible minimum versions. |
| 9 | The clustering scatter draws all 100k points, which is slow. | Use a stratified random sample (about 3,000 points) for scatter plots. |
| 10 | Bare `except:` blocks swallow errors, and the app calls `sys.exit` inside methods. | Narrow the exceptions; use a clear error dialog on load failure. |

**Builder's call:** the 18 "Other"-gender rows are too few to model meaningfully. They are excluded from training, and the gender input offers Female / Male. The README notes this.

## Architecture

```
analysis/                 # shared Python package (new)
  data.py                 # load, dedupe, encode → clean DataFrame + feature list
  models.py               # train RF + LR, compute metrics, ROC, confusion matrices
  clustering.py           # K-means with stable label ordering + generated descriptions
scripts/
  export_web.py           # runs the pipeline, writes web/data/*.json
web/                      # the deployable folder — copy this to the portfolio site
  index.html
  styles.css
  js/*.js                 # vanilla ES modules, no build step
  data/                   # generated JSON (committed, so the folder works as-is)
tests/                    # pytest: pipeline + JS/Python prediction parity
main.py                   # desktop app, refactored to use analysis/
```

Both the desktop app and the web export use the same `analysis/` code, so the fixes live in one place.

### Model export

- **Logistic Regression:** export the scaler means and scales, the coefficients and the intercept. The JS computes the sigmoid. The LR is wrapped in a `StandardScaler` pipeline, which also makes it converge properly.
- **Random Forest:** export each tree as compact arrays (feature, threshold, left, right, leaf probability). The JS walks the trees and averages them. Expect roughly 150 trees × up to 255 nodes, or about 1 MB of JSON. If it comes out larger, the builder may cut the tree count to 100, provided the AUC drops by less than 0.005.
- **Parity test:** `export_web.py` also writes about 200 test inputs with Python's predicted probabilities. A Node test (`node --test`) loads the JSON, runs the JS model code, and asserts each probability matches within 1e-6.

### Exported page data

Everything is precomputed in Python, so the browser does no heavy lifting:
- per-feature histograms (bins and counts), split by diabetic vs not
- category counts for gender, smoking, hypertension and heart disease
- the correlation matrix
- cluster centres, sizes, diabetes rates, generated descriptions, and a scatter sample
- each model's metrics, ROC curve points (downsampled to about 100), and confusion matrix
- RF feature importances and LR coefficients (for a "what drives the prediction" view)
- dataset summary: row count, duplicates removed, class balance

## The page

A single scrolling page with a sticky section nav. It works at phone width and supports light and dark themes, following the system with a toggle.

```
┌───────────────────────────────────────────────────────────────┐
│ Diabetes Risk Explorer        Explore Clusters Models Try it  │
├───────────────────────────────────────────────────────────────┤
│ 100k records · 2 models · best AUC 0.97   [Try the predictor] │
│ one-paragraph intro + "educational, not medical advice"       │
├─ Explore ─────────────────────────────────────────────────────┤
│ [feature picker ▾]  histogram, diabetic vs non-diabetic       │
│ correlation heatmap (hover for values) + 4 hypotheses w/ src  │
├─ Clusters ────────────────────────────────────────────────────┤
│ scatter HbA1c × BMI coloured by cluster · cluster cards       │
│ (size, avg age/BMI/HbA1c, diabetes rate, generated summary)   │
├─ Models ──────────────────────────────────────────────────────┤
│ metrics table (best per row highlighted) · ROC curves         │
│ confusion matrices side by side · feature importance          │
├─ Try it ──────────────────────────────────────────────────────┤
│ gender · age · BMI · HbA1c · glucose · hypertension ·         │
│ heart disease · smoking     →  live risk gauge per model      │
│ (both models shown together, updates as you move sliders)     │
│ disclaimer                                                    │
├─ Findings & limitations ──────────────────────────────────────┤
│ links: GitHub repo · original Gatehouse version               │
└───────────────────────────────────────────────────────────────┘
```

**Builder's calls:**
- Charts use Observable Plot (pinned version from a CDN). It is light, handles heatmaps and histograms well, and is easy to theme.
- The predictor shows **both** models' probabilities at once, instead of a radio button. Seeing the models disagree is more interesting.
- The risk bands stay as Low < 40% ≤ Moderate < 70% ≤ High. They are labelled as model probability, not clinical risk.
- Slider ranges come from the data's 1st–99th percentiles, with the full range reachable by typing a value.

## Testing

- pytest: deduplication, encoding (no ordinal smoking), stable cluster ordering across seeds, metrics sanity checks (AUC > 0.9), export schema.
- Node parity test (described above).
- A Playwright smoke test: the page loads from `web/` over a local static server, every section renders, and moving a slider changes the probability.

## Delivery

- Work on branch `ccr-b127e3e2-08fblo` and push there.
- The README gets the new structure, how to regenerate `web/data`, how to deploy the `web/` folder, and screenshots.
- Placing the folder onto Tsakioan.com is the owner's step. The README gives instructions for common hosts.

## Left to the builder

Exact visual styling (it should be clean and readable, not templated), JS module boundaries, histogram bin counts, and copy wording beyond the facts above.
