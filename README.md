# Explainable AI for Disease Classification: A Reproducible ML Benchmark

A research-oriented, reproducible Python experiment comparing logistic regression and random forest on the **Wisconsin Diagnostic Breast Cancer** dataset available through scikit-learn. The project focuses on rigorous train/test separation, cross-validation, interpretable feature importance, automated tests, and an interactive demonstration.

> **Safety:** This is a classroom/research benchmark, **not** a medical diagnosis tool. Dataset predictions are not validated for real patients.

## Research question

How do a linear baseline and a tree ensemble compare on a fixed tabular classification benchmark, and which features are most influential under permutation importance?

## Method

1. Load the 569-example, 30-feature Wisconsin Diagnostic Breast Cancer benchmark through scikit-learn.
2. Recode **malignant as positive class 1** for easier interpretation.
3. Make a stratified 80/20 training/test split with random seed 42.
4. Compare two classifiers using stratified 5-fold cross-validation **on the training set only**.
5. Select the highest mean cross-validation ROC AUC model.
6. Evaluate that selected model once on the untouched held-out test set.
7. Compute held-out permutation importance and export a plot, CSV, and machine-readable metrics.

## Setup

Requires Python 3.10+ (Python 3.11 recommended).

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
python -m src.train
python -m pytest -q
streamlit run app.py
```

## Outputs

Training writes these files to `artifacts/`:

- `metrics.json`: training cross-validation scores and held-out test scores
- `feature_importance.csv`: ranked permutation importances
- `feature_importance.png`: visualization
- `model.joblib`: trained model (do not load untrusted model files)

**Results are intentionally not invented.** Run the experiment and cite the actual `metrics.json` in any application or report.

## Project structure

```text
app.py
src/train.py
src/__init__.py
tests/test_pipeline.py
.github/workflows/ci.yml
requirements.txt
README.md
LICENSE
.gitignore
```

## Limitations and future research

- Small historical benchmark; it does not establish external validity, clinical utility, or fairness across populations.
- No independent external validation cohort or calibration study.
- Permutation importance measures association with predictive performance, **not causality**; correlated features can complicate interpretation.
- A single held-out split yields uncertain performance estimates; consider repeated nested cross-validation and confidence intervals.
- The Streamlit interface explores existing dataset records only; it does not collect patient information.

## Suggested extensions

- Add nested cross-validation and bootstrap confidence intervals.
- Compare permutation importance stability across folds.
- Add probability calibration curves and Brier scores.
- Reproduce results across several random seeds and publish a methods/results table.

## Dataset and attribution

Dataset: *Breast Cancer Wisconsin (Diagnostic)*, original contributors W. N. Street, W. H. Wolberg, and O. L. Mangasarian, distributed via the scikit-learn datasets module. Consult the scikit-learn dataset documentation for source and citation details.

## Responsible presentation

Only claim authorship for work you personally understand and have tested. Clearly distinguish provided starter code from your own extensions in internship applications.

## License

MIT.
