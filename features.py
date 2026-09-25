"""Feature engineering shared by the training notebook and the Streamlit app.

Keeping this in one place guarantees the app builds exactly the same inputs
the model was trained on.
"""
import numpy as np
import pandas as pd

BINDERS = ["OPC", "OPC+FA", "PPC"]
FEATURES = ["fly_ash", "ppc", "ld_slag_pct", "log_age"]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Turn (binder, ld_slag_pct, age_days) into model inputs.

    - binder is one-hot encoded with OPC as the reference level
    - age enters as log(age): strength gain slows with time, so a log scale
      makes the age effect close to linear
    """
    unknown = set(df["binder"]) - set(BINDERS)
    if unknown:
        raise ValueError(f"Unknown binder type(s): {unknown}. Use one of {BINDERS}.")
    return pd.DataFrame(
        {
            "fly_ash": (df["binder"] == "OPC+FA").astype(float),
            "ppc": (df["binder"] == "PPC").astype(float),
            "ld_slag_pct": df["ld_slag_pct"].astype(float),
            "log_age": np.log(df["age_days"].astype(float)),
        },
        index=df.index,
    )[FEATURES]
