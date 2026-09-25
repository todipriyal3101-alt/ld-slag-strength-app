# 🧱 LD Slag Concrete Strength Predictor

A machine learning model that predicts the cube compressive strength (MPa) of M40 concrete
from binder type, LD (Linz-Donawitz steel) slag content and curing age, trained on lab test results.

## Dataset

`data/ld_slag_strength.csv`: 13 mixes × 4 ages (7, 28, 56, 90 days) × 3 cubes (150 mm) = 156 results.

| Column | Meaning |
|---|---|
| mix | Mix ID (MOPL0, MOFL0–10, MPPL0–10) |
| binder | `OPC` (OPC 53 control), `OPC+FA` (OPC 53 + fly ash), `PPC` |
| ld_slag_pct | LD slag content, % |
| age_days | Curing age, days |
| specimen | Cube number (1–3) |
| strength_mpa | Compressive strength, MPa |

Target mean strength (IS 10262, M40): 32.33 MPa at 7 days, 48.25 MPa at 28 days.

## Methodology

1. **Data checks:** cube spread against the IS 456 ±15 % rule, and a scan for identical triplets across mixes.
2. **Features:** binder (one-hot), LD slag %, log(age).
3. **Evaluation:** leave-one-mix-out cross-validation. Every fold holds out all 12 results of one mix, so the model is always scored on a dosage it has never seen. A random split would leak cubes of the same mix into both sets.
4. **Baseline:** a slag-blind model (average of other mixes with the same binder and age). R² is high for every model because age dominates, so RMSE against this baseline is the meaningful comparison.
5. **Models:** Linear Regression, Polynomial + Ridge, SVR, Random Forest, XGBoost (untuned and GridSearchCV-tuned), Gaussian Process.
6. **Final model:** Gaussian Process, which gives a smooth dosage curve and a 95 % range with each prediction.
7. **Interpretability:** SHAP on the tuned XGBoost model.

## Results (leave-one-mix-out)

| Model | R² | RMSE (MPa) | MAE (MPa) |
|---|---|---|---|
| XGBoost (tuned)* | 0.992 | 0.84 | 0.65 |
| **Gaussian Process (final)** | **0.992** | **0.85** | **0.66** |
| XGBoost (untuned) | 0.990 | 0.93 | 0.71 |
| Polynomial (deg 3) + Ridge | 0.990 | 0.95 | 0.71 |
| Random Forest | 0.989 | 0.98 | 0.76 |
| Baseline (ignores slag) | 0.981 | 1.27 | 1.06 |
| SVR | 0.980 | 1.32 | 0.99 |
| Linear Regression | 0.935 | 2.39 | 1.95 |

\*Tuned on the same folds, so slightly optimistic. Replicate noise between cubes is about 0.44 MPa, which is the floor no model can beat.

Predicted optimum LD slag at 28 days: about 6 % for OPC + fly ash and about 7.5 % for PPC.

## Limitations

The model is a smooth fit to one experiment, not a general concrete strength predictor. It is valid only for 0–10 % LD slag, 7–90 days, these three binders and this M40 mix design. Fly ash content and what LD slag replaces are not inputs. Adding each mix's kg/m³ proportions to the CSV would let the model use them.

## Running locally

```bash
pip install -r requirements-notebook.txt
cd notebooks && jupyter notebook ld_slag_strength_model.ipynb   # retrains and saves the model
cd .. && streamlit run app.py
```

## Repository structure

```
├── app.py                          # Streamlit app
├── features.py                     # Feature code shared by notebook and app
├── ld_slag_strength_model.pkl      # Trained model + metadata
├── model_results_summary.csv       # Cross-validation results
├── data/ld_slag_strength.csv
├── notebooks/ld_slag_strength_model.ipynb
├── requirements.txt                # App dependencies (used by Render)
├── requirements-notebook.txt       # Extra dependencies for retraining
└── render.yaml
```
