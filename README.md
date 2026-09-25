# 🧱 LD Slag Concrete Strength Predictor

**Live app:** https://ld-slag-strength-app.onrender.com

A machine learning model that predicts the 150 mm cube compressive strength (MPa) of M40 concrete from binder type, LD (Linz-Donawitz steel) slag content and curing age, trained on lab test results.

## Dataset

`LD DATA SET.csv`: lab results for 13 mixes, each tested at 7, 28, 56 and 90 days with 3 cubes per age (156 results).

| Mix | Binder | LD slag |
|---|---|---|
| MOPL0 | OPC 53 (control) | 0 % |
| MOFL0 – MOFL10 | OPC 53 + fly ash | 0, 2, 4, 6, 8, 10 % |
| MPPL0 – MPPL10 | PPC | 0, 2, 4, 6, 8, 10 % |

Target mean strength (IS 10262, M40): 32.33 MPa at 7 days and 48.25 MPa at 28 days.

## Methodology

1. **Data preparation:** the lab table is reshaped into one row per cube (mix, binder, LD slag %, age, strength).
2. **Features:** binder type (fly ash / PPC flags), LD slag %, and log(age).
3. **Evaluation:** leave-one-mix-out cross-validation. Each mix is predicted by a model trained on the other 12, so every score is for a mix the model has never seen.
4. **Models compared:** Linear Regression, Random Forest, XGBoost and Gaussian Process.
5. **Final model:** Gaussian Process, which gives a smooth curve between tested dosages and a 95 % range with each prediction.

## Results (leave-one-mix-out)

| Model | R² | RMSE (MPa) | MAE (MPa) |
|---|---|---|---|
| XGBoost | 0.989 | 0.97 | 0.69 |
| **Gaussian Process (final)** | **0.989** | **0.98** | **0.70** |
| Random Forest | 0.985 | 1.13 | 0.81 |
| Linear Regression | 0.934 | 2.38 | 1.95 |

R² is high for every model because curing age drives most of the strength change, so RMSE (about 1 MPa) is the more meaningful measure.

**Predicted optimum LD slag at 28 days:** 6.5 % for OPC + fly ash (about 50.6 MPa) and 7.5 % for PPC (about 51.6 MPa).

## Limitations

The model is fitted to one experiment and is valid only for 0–10 % LD slag, 7–90 days, these three binders and this M40 mix design. Fly ash content and the material LD slag replaces are not model inputs.

## Running locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

To retrain the model, open `RP_2.ipynb` in Jupyter and run all cells.

## Files

```
├── app.py                      # Streamlit app
├── features.py                 # Input features used by the app
├── ld_slag_strength_model.pkl  # Trained model
├── LD DATA SET.csv             # Lab test data
├── RP_2.ipynb                  # Training notebook
├── requirements.txt            # App dependencies (used by Render)
└── render.yaml                 # Render deployment settings
```
