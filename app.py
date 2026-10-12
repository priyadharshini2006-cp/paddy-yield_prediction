import json
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Paddy Yield Prediction", page_icon="🌾", layout="wide")

@st.cache_resource
def load():
    model = joblib.load("yield_model.pkl")
    meta = json.load(open("model_meta.json"))
    return model, meta

model, meta = load()
ranges, choices = meta["ranges"], meta["choices"]

st.title("🌾 Paddy Yield Prediction")
st.caption("Enter farm details to estimate paddy yield. Model: "
           f"{meta['best']} (test R² ≈ {meta['results'][meta['best']]['R2']:.2f}, "
           f"MAE ≈ {meta['results'][meta['best']]['MAE']:.0f} kg/ha)")

def num_input(col, container):
    lo, hi, med = ranges[col]
    step = 1.0 if float(med).is_integer() and hi - lo > 20 else 0.1
    return container.number_input(col, min_value=lo, max_value=hi, value=med, step=step, key=col)

MAIN = ["Hectares", "Seedrate(in Kg)", "LP_Mainfield(in Tonnes)", "Nursery area (Cents)",
        "LP_nurseryarea(in Tonnes)", "DAP_20days", "Weed28D_thiobencarb", "Urea_40Days",
        "Potassh_50Days", "Micronutrients_70Days", "Pest_60Day(in ml)", "Trash(in bundles)"]

vals = {}
st.subheader("Field details")
c = st.columns(4)
for i, col in enumerate(meta["cat"][:4]):          # Agriblock, Variety, Soil Types, Nursery
    vals[col] = c[i].selectbox(col, choices[col], key=col)

st.subheader("Inputs & fertilizer")
c = st.columns(4)
for i, col in enumerate([m for m in MAIN if m in ranges]):
    vals[col] = num_input(col, c[i % 4])

with st.expander("Weather details (defaults = dataset average)"):
    c = st.columns(4)
    rest = [x for x in meta["columns"] if x not in vals]
    for i, col in enumerate(rest):
        if col in choices:
            vals[col] = c[i % 4].selectbox(col, choices[col], index=0, key=col)
        else:
            vals[col] = num_input(col, c[i % 4])

if st.button("Predict yield", type="primary"):
    row = pd.DataFrame([vals])[meta["columns"]]
    per_ha = float(model.predict(row)[0])
    total = per_ha * vals["Hectares"]
    a, b = st.columns(2)
    a.metric("Predicted yield per hectare", f"{per_ha:,.0f} kg/ha")
    b.metric(f"Total for {vals['Hectares']:g} ha", f"{total:,.0f} kg")
    mae = meta["results"][meta["best"]]["MAE"]
    st.info(f"Typical error of this model is about ±{mae:.0f} kg/ha, so treat this as an estimate.")
