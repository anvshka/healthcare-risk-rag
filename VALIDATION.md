# Stage 1 validation

Validated on 26 October 2026 with Python 3.14.2, pandas 3.1.0 and pytest 9.1.1.

```text
python -m pytest -q
............................                                             [100%]
28 passed in 1.22s
```

Tests cover all recognized datetime columns, invalid datetime/numeric input, calendar age and shifted-age capping, missing/negative ages, conflicting birth dates, stripping/uppercasing, UNKNOWN categories, the strictly-greater-than-60% missingness boundary, constant-column removal, protected keys/target/lab values, deduplication before median estimation, per-item lab medians, groups without observations, idempotence, input immutability, leading-zero codes, normalized headers, ambiguous files, empty dated tables, schema-only CLI output/report writing, and the missing-file error path.

The empty-dated-table edge test initially exposed a pandas empty-Series mapping error. The age join now uses indexed reindexing, and all 28 tests pass after that correction.

The actual Stage 1 CLI was also run from the repository root:

```text
python -m src.preprocessing
Missing MIMIC-III files in .../healthcare-risk-rag/data/raw:
  PATIENTS.csv (or .csv.gz)
  ADMISSIONS.csv (or .csv.gz)
  ICUSTAYS.csv (or .csv.gz)
  DIAGNOSES_ICD.csv (or .csv.gz)
  LABEVENTS.csv (or .csv.gz)
  D_LABITEMS.csv (or .csv.gz)
  PRESCRIPTIONS.csv (or .csv.gz)
Place your authorized MIMIC-III Clinical Database DEMO CSVs there and rerun. No data is downloaded or generated.
```

It exited with code 1 as intended. No processed dataset or real-data preprocessing report was generated. The tests use only isolated in-memory examples and temporary schema files, never a replacement dataset in data/raw.

## Real-data validation after dataset supplied

Stage 1 was rerun successfully on 26 October 2026 after the user supplied all seven CSVs. The earlier missing-file result above records the original check before the dataset was supplied.

- Test suite: **28 passed in 1.53s**.
- Preprocessing CLI: exit code **0**; seven processed CSVs and `reports/preprocessing_report.md` saved.
- Real-data idempotence: exact pandas frame equality passed for all seven tables after a second preprocessing call.
- Saved CSV row counts and headers: verified against all seven in-memory cleaned tables.
- Patients: 100; admissions: 129; ICU stays: 136; diagnoses: 1,761; lab events: 76,074; lab dictionary: 753; prescriptions: 10,398. All input row counts were retained.
- Age: no missing admission ages; observed range 17-90 years.
- Numeric labs: 6,896 values remain missing because their item has no observed numeric median. This is intentional; no cross-item substitution is made.
- Removed columns: `patients.expire_flag` (constant), `admissions.deathtime` (>60% missing), `labevents.flag` (>60% missing).

The percentage of missing cells in patients rises from 6.62% to 7.57% because removing a fully observed constant column reduces the denominator; no new patient missing values were introduced.

Status: Stage 1 tests and actual-data preprocessing pass. Stage 2 has not been started. The distributable ZIP excludes raw and processed datasets and generated reports.


## Stage 2 validation

Validated on 26 October 2026 with the same Python/pandas/pytest environment.

```text
python -m pytest -q
.......................................................                  [100%]
55 passed in 3.36s
```

The 28 Stage 1 tests remain passing. The additional 27 Stage 2 cases cover one row per admission, relationship ownership, counts, lab summaries and training-only medians, training-only top-lab selection, population z-scores, constant features, absent events, missing categorical values, no direct outcome-field/identifier leakage, invariance to held-out measurement changes, patient-disjoint deterministic splits, JSON state round trips, input immutability, invalid labels/dates/IDs, numeric/string ID compatibility, infinite lab observations, missing files, and CLI artifact alignment.

The actual `python -m src.features` run exited with code 0:

- 129 admissions and 100 unique patients.
- Training: 104 admissions / 80 patients (69 survival, 35 mortality labels).
- Testing: 25 admissions / 20 patients (20 survival, 5 mortality labels).
- Zero patient overlap.
- 38 predictors: 35 numeric and 3 categorical.
- Ten selected labs: potassium, hematocrit, sodium, chloride, bicarbonate, glucose, creatinine, urea nitrogen, anion gap and platelet count.
- Seven Stage 2 output files plus `reports/feature_report.md` saved.

An independent saved-artifact audit verified 129 unique admission IDs, split alignment, exact feature-column order, finite numeric values in training/testing, zero training means and unit population standard deviations for nonconstant columns. Restoring the JSON state reproduces the saved training matrix within floating-point tolerance.

No new dependencies were added. Stage 2 reloads raw observations and does not reuse the full-data lab medians from Stage 1. Its full-stay predictors are documented as retrospective. The Stage 2 development split is provisional; GroupShuffleSplit, categorical encoding, model fitting and fold-wise refitting remain Stage 3 work.

Status: Stages 1 and 2 are implemented and tested successfully on the real dataset. Stage 3 has not been started. Code-only archives exclude all raw/processed datasets, fitted state and generated reports.


## Stage 3 validation

Stage 3 completed on 26 October 2026. scikit-learn and matplotlib were explicitly approved by the user; XGBoost was already requested in the project brief. Tested versions: scikit-learn 1.9.1, XGBoost 3.4.1, matplotlib 3.11.2, pandas 3.1.0, NumPy 2.5.4, Python 3.14.2 and pytest 9.1.1.

```text
python -m pytest -q
........................................................................ [ 92%]
......                                                                   [100%]
78 passed in 9.57s
```

All 55 existing cases remain passing. The 23 additional Stage 3 cases cover exact metric values/confusion matrices, single-class evaluation, invalid probabilities/outcomes, threshold behavior, reproducible grouped holdout, five grouped CV folds limited to training patients, infeasible CV failure, class weighting, training-only encoding, invariance to held-out measurement changes, rejection of patient overlap and row misalignment, pickle round trips, prediction-ID alignment, CV model selection, missing-data errors and PNG plot outputs.

The actual `python -m src.train` run completed all five CV folds and exited with code 0. It trained all three requested models, using the fixed parameters saved in model_metrics.json.

- Training: 80 patients / 107 admissions (77 surviving, 30 mortality labels).
- Holdout: 20 patients / 22 admissions (12 surviving, 10 mortality labels).
- Patient overlap: zero in holdout and every CV fit/validation partition.
- Every training admission has exactly one out-of-fold prediction per model (107 per candidate); no held-out admission appears in CV predictions.
- Random Forest was selected by mean training-CV average precision, before evaluating the holdout.
- Held-out ROC-AUC: Logistic Regression 0.642, Random Forest 0.650, XGBoost 0.592.
- Random Forest held-out average precision 0.758, precision 1.000, recall 0.500 and F1 0.667; confusion matrix [[12, 0], [5, 5]].
- Random Forest training-CV ROC-AUC 0.796 +/- 0.101 and average precision 0.698 +/- 0.153.

An independent artifact audit verified patient isolation using every saved fold's fitted IDs, complete unique OOF coverage, final preprocessing fitted exclusively on training IDs, and CV-only model selection. All three saved pickle bundles reproduced the held-out probabilities and metrics within floating-point tolerance. Loading the native XGBoost JSON also reproduced its held-out probabilities after applying the saved encoder.

ROC, precision-recall and confusion-matrix PNGs were visually inspected for readable labels and consistency with metrics. A matplotlib home-directory font-cache warning was corrected by keeping the cache within reports/.matplotlib; plots were regenerated successfully.

Stage 3 is complete. These results are a small retrospective benchmark, with substantial uncertainty and no clinical validity claim. Stage 4 has not been started. The updated archive excludes raw/processed data, fitted models, patient-level audits and generated reports.


## Stage 4 validation

Stage 4 completed on 26 October 2026 using SHAP 0.54.1 in the same recorded Python/scikit-learn/pandas environment. SHAP was explicitly requested in the original brief; no additional unrelated package was added.

```text
python -m pytest -q
........................................................................ [ 73%]
..........................                                               [100%]
98 passed in 14.46s
```

The 78 prior cases remain passing. The additional 20 cases cover binary class-1/scalar output shapes, invalid/nonfinite SHAP output, probability additivity, complete categorical dummy grouping, held-out reference isolation, top-five ranking/signs/magnitudes, original-unit values, missing-value indicators, bad admission queries, invalid row positions, no fitted-state changes, saved encoder consistency, missing training data, Logistic Regression and XGBoost permutation fallbacks, missing model/manifest/data failures, generated artifacts and explicit rejection of a failed reconstruction.

The initial probability-space interventional Tree SHAP smoke check in this installed stack did not reconstruct Random Forest probabilities at the chosen tolerance. The implementation uses the model's fitted training-path Tree SHAP instead, with an explicit probability reconstruction check. No SHAP values or baseline are manually adjusted to force agreement.

The actual `python -m src.explain` run exited with code 0:

- Selected model: Random Forest; no model, scaler, imputer or encoder was refitted.
- 22 held-out admissions explained; 107 admissions from 80 training patients supply the fitted tree statistics.
- 44 encoded features, grouped into 38 original features.
- Maximum probability reconstruction error: 4.996003610813204e-16.
- All 22 records have five leading factors, their signed contribution/magnitude and the remainder needed for full reconstruction.
- Three PNGs saved: global beeswarm, grouped global importance and an individual top-five plot. All three were visually inspected for readable labels and coherent output units.
- Global ranking led by mean bicarbonate, length of stay, maximum bicarbonate and mean anion gap; these are model attributions, not clinical or causal conclusions.

An independent audit verified that every saved explanation's admission ID and probability matched the Stage 3 held-out predictions. No training patient was included in the global summary. The saved encoded and grouped contribution sums agreed, baseline plus full contributions reconstructed each prediction, top-five signs/magnitudes and membership matched the full grouped table, and mean absolute grouped importance matched the saved global ranking.

The Random Forest baseline is 0.5 under its fitted training-path/class-weight statistics, not the empirical mortality rate. Other-model probability-space permutation fallbacks were tested on isolated fixtures; the actual-data run uses Random Forest. Whole-stay and small-cohort limitations still apply. Stage 4 explains the model rather than supplying clinical evidence.

Status: Stages 1-4 are implemented and tested successfully. Stage 5 has not been started. The updated ZIP excludes patient data, fitted bundles, audits and generated reports.


## Stage 5 validation

Completed on 27 October 2026 (Asia/Calcutta). Tested additions: sentence-transformers 5.7.0, faiss-cpu 1.15.1, PyTorch 2.14.1, transformers 5.20.2 and huggingface-hub 2.3.0, alongside the prior recorded runtime. The requested all-MiniLM-L6-v2 embedding model was downloaded to the workspace cache; no patient records were downloaded or uploaded.

```text
python -m pytest -q
........................................................................ [ 54%]
............................................................             [100%]
132 passed in 25.93s
```

All 98 earlier cases remain passing. The 34 Stage 5 cases cover source metadata/provenance, placeholder relabeling, URL/access requirements, exact chunk offsets and token budgets, overlapping coverage, cosine search, deterministic cache reuse, source/model revision invalidation, corrupted-index rejection, invalid vectors, readable feature queries/labels, exact citations, local templates, no-source abstention, placeholder-only behavior, sanitized provider payloads, rejected invented claims/citations/altered excerpts, provider outages, source-instruction handling, complete-quote preservation, invalid SHAP facts, unsupported providers/endpoints, HTTP serialization, missing Stage 4 input, saved outputs/admission filtering and valid provider abstention without mislabeling literature.

Small deterministic vector/text fixtures are used only in unit tests. The production integration run used the actual pretrained model and FAISS; no test embedder or lexical/synthetic embedding fallback exists in production.

Actual run: `python -m src.rag --download-model` completed successfully. A second run used cached weights with HF_HUB_OFFLINE=1, TRANSFORMERS_OFFLINE=1, LLM_PROVIDER=none and no download flag; it also exited with code 0.

- 22 saved Stage 4 admission records processed.
- Five nonclinical placeholder files produced five knowledge chunks.
- Actual embedding dimensions: 384.
- Embedding model revision: 1110a243fdf4706b3f48f1d95db1a4f5529b4d41.
- Status counts: placeholder_only=22, literature_retrieved=0, no_relevant_source=0.
- No LLM service or key was used. Placeholder-only data does not trigger the optional HTTP provider.
- FAISS index and manifest, all JSON explanations, a readable Markdown example, metadata and a Stage 5 report were saved.

An independent offline audit loaded the cached model/index, checked actual retrieval of the acid-base placeholder for bicarbonate/anion-gap terms, and compared FAISS scores with explicit cosine dot products of its stored real vectors. It verified all 22 admission IDs against Stage 4, every cited source's content hash and exact chunk offsets/text, citation membership, placeholder-only labels, absence of accepted medical passages and presence of the model-source citation/disclaimer.

Only placeholder knowledge has been supplied. The plumbing, no-API mode and citation/claim-validation boundaries are tested; real literature retrieval quality, clinical correctness and applicability are not validated. HTTP selection was tested using injected responses and a mocked transport, not a live external LLM. The optional provider can select only complete retrieved excerpts; free-form clinical sentences are not accepted.

Status: Stage 5 implementation and integration tests pass. Real medical grounding requires user-reviewed literature in data/knowledge. Stage 6 has not been started. The code ZIP includes the five original placeholder files and their instructions, but excludes patient data, model bundles, vector indexes, downloaded model caches, audits and generated explanations.


## Stage 6 validation

Completed on 27 October 2026 (Asia/Calcutta) with Streamlit 1.66.1 and the previously recorded runtime. Streamlit was already requested by the project brief. Its transitive UI/runtime dependencies installed in the workspace environment.

```text
python -m pytest -q
........................................................................ [ 49%]
........................................................................ [ 99%]
.                                                                        [100%]
145 passed in 67.67s (0:01:07)
```

All 132 earlier cases remain passing. The 13 Stage 6 cases cover current catalog/scope and outcome exclusion, frozen model state, saved-context reuse without embedding downloads, attribution/identity/provider/source provenance, knowledge-edit invalidation, local-only live retrieval, unknown admissions, AppTest default score/placeholder rendering, patient/admission switching and training warnings, missing-project setup, missing-retrieval recovery without hiding scores, loopback/telemetry settings and readable non-mojibake UI text.

The Stage 5 response schema gained exact SHAP-record hashes and index fingerprints so the app can reject stale narratives. Stage 5 outputs were regenerated offline with no API key. No model was retrained.

An independent real-data audit verified:

- Current catalog: 129 admissions / 100 patients; default held-out scope: 22 admissions / 20 patients.
- Three selected held-out score/SHAP outputs agreed with saved Stage 4 predictions.
- A training admission was explicitly marked training and reconstructed its model output within tolerance.
- Current attribution hashes matched cited-context hashes.
- Real on-demand retrieval for a training admission succeeded with cached embedding weights, no API service and placeholder-only status.
- The running server's http://127.0.0.1:8501/_stcore/health endpoint returned `ok`.

Browser verification showed a real default admission score of 27.3%, a 50.0% model reference, the signed factor chart/table and placeholder notice. Changing the selected patient updated the admission, score (77.9% for the next checked admission), factor values and explanation consistently. The browser was left open as a local deliverable. An inspection caught narrow-column stacking and UTF-8 punctuation issues; the layout/source encoding were corrected and all tests rerun successfully. A preview image was saved outside the distributable repository archive.

The initial detached background start did not remain running in the managed shell. The server was then started in a persistent execution session and verified through both HTTP and the browser. The companion local Windows launcher provides a restart path using the original prepared environment; portable setup remains in README.md/START_HERE.md.

Status: all six software stages are implemented and tested. Real medical-literature context remains pending because the knowledge folder still contains five placeholders. No public deployment, API-key requirement, fabricated dataset or clinical-validity claim was introduced. The final code ZIP excludes patient data, trained artifacts, generated explanations, indexes, caches and screenshots.
