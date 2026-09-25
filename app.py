import altair as alt
import joblib
import numpy as np
import pandas as pd
import streamlit as st

from features import build_features

st.set_page_config(page_title="LD Slag Concrete Strength", page_icon="🧱", layout="centered")


@st.cache_resource
def load_bundle():
    return joblib.load("ld_slag_strength_model.pkl")


bundle = load_bundle()
model = bundle["model"]
measured = pd.DataFrame(bundle["measured_means"])
slag_min, slag_max = bundle["training_ranges"]["ld_slag_pct"]
age_min, age_max = bundle["training_ranges"]["age_days"]


def predict(binder, slag, age):
    frame = pd.DataFrame({"binder": binder, "ld_slag_pct": slag, "age_days": age})
    mean, sd = model.predict(build_features(frame), return_std=True)
    return mean, sd


st.title("🧱 M40 concrete with LD slag: strength predictor")
st.write(
    "Predicts 150 mm cube compressive strength from binder type, LD slag content and "
    "curing age. Trained on 13 lab mixes tested at 7, 28, 56 and 90 days."
)

# ---------- Inputs ----------
binder_labels = {"OPC 53 + fly ash": "OPC+FA", "PPC": "PPC", "OPC 53 only (control, 0 % slag)": "OPC"}
choice = st.radio("Binder", list(binder_labels), horizontal=True)
binder = binder_labels[choice]

col1, col2 = st.columns(2)
with col1:
    if binder == "OPC":
        slag = 0.0
        st.slider("LD slag (%)", slag_min, slag_max, 0.0, disabled=True,
                  help="Only the 0 % OPC control was tested, so slag can't be varied for OPC.")
    else:
        slag = st.slider("LD slag (%)", slag_min, slag_max, 6.0, step=0.5)
with col2:
    age = st.slider("Curing age (days)", age_min, age_max, 28, step=1)

# ---------- Prediction ----------
mean, sd = predict([binder], [slag], [age])
mean, sd = float(mean[0]), float(sd[0])
low, high = mean - 1.96 * sd, mean + 1.96 * sd

st.metric("Predicted compressive strength", f"{mean:.1f} MPa")
st.caption(f"95 % range for a single cube: {low:.1f} – {high:.1f} MPa")

target = bundle["targets"].get(str(age))
if target is not None:
    verdict = "meets" if mean >= target else "falls below"
    st.write(f"This {verdict} the {age}-day target mean strength of {target} MPa.")

tested = measured[(measured.binder == binder) & (measured.ld_slag_pct == slag) & (measured.age_days == age)]
if not tested.empty:
    st.write(f"Measured average for this exact mix and age: **{tested.strength_mpa.iloc[0]:.2f} MPa**.")

# ---------- Dosage curve ----------
if binder != "OPC":
    st.subheader(f"Strength vs LD slag at {age} days")
    grid = np.arange(slag_min, slag_max + 0.01, 0.25)
    g_mean, g_sd = predict([binder] * len(grid), grid, [age] * len(grid))
    curve = pd.DataFrame({"ld_slag_pct": grid, "pred": g_mean,
                          "low": g_mean - 1.96 * g_sd, "high": g_mean + 1.96 * g_sd})
    best = curve.loc[curve.pred.idxmax()]

    x = alt.X("ld_slag_pct:Q", title="LD slag (%)")
    band = alt.Chart(curve).mark_area(opacity=0.2).encode(x=x, y=alt.Y("low:Q", title="Strength (MPa)",
                                                                             scale=alt.Scale(zero=False)), y2="high:Q")
    line = alt.Chart(curve).mark_line().encode(x=x, y="pred:Q")
    pts = measured[(measured.binder == binder) & (measured.age_days == age)]
    layers = [band, line]
    if not pts.empty:
        layers.append(alt.Chart(pts).mark_point(filled=True, size=70, color="black").encode(
            x="ld_slag_pct:Q", y="strength_mpa:Q", tooltip=["mix", alt.Tooltip("strength_mpa:Q", format=".2f")]))
    layers.append(alt.Chart(pd.DataFrame([best])).mark_point(shape="diamond", size=160, color="red").encode(
        x="ld_slag_pct:Q", y="pred:Q"))
    st.altair_chart(alt.layer(*layers), use_container_width=True)
    st.write(f"Predicted optimum: **{best.ld_slag_pct:.2f} % LD slag**, about {best.pred:.1f} MPa. "
             "Black dots are measured averages where this age was tested; the shaded band is the 95 % range.")

with st.expander("About this model"):
    cv = pd.DataFrame(bundle["cv_scores"]).T.sort_values("RMSE")
    st.write(
        f"Model: {bundle['model_name']}. Scores below are leave-one-mix-out cross-validation, "
        "i.e. each mix is predicted by a model that never saw it."
    )
    st.dataframe(cv)
    st.write(
        "Valid only for 0–10 % LD slag, 7–90 days, these binders and this M40 mix design. "
        "It is a fit to one experiment, not a general concrete strength predictor."
    )
