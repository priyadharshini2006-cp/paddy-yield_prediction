"""
Prediction and testing of the deployed model (files from 7_Model_Deployment).

Part A: scenario tests (typical, low/high inputs, different varieties, edge cases)
Part B: batch test on 25 real records from the dataset (actual vs predicted)
"""
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
DEPLOY = BASE.parent / "7_Model_Deployment"
DATA = BASE.parent / "1_Dataset_Assigned" / "paddydataset.csv"

model = joblib.load(DEPLOY / "yield_model.pkl")
meta = json.load(open(DEPLOY / "model_meta.json"))
cols, ranges, choices = meta["columns"], meta["ranges"], meta["choices"]

# Baseline farm = dataset median for numbers, first category for text
base = {c: ranges[c][2] for c in ranges}
base.update({c: choices[c][0] for c in choices})


def predict(**changes):
    row = {**base, **changes}
    per_ha = float(model.predict(pd.DataFrame([row])[cols])[0])
    return per_ha, per_ha * row["Hectares"]


def lo(c): return ranges[c][0]
def hi(c): return ranges[c][1]


tests = [
    ("T1", "Typical farm (all defaults)", {}),
    ("T2", "High fertilizer (max Urea and Potash)", {"Urea_40Days": hi("Urea_40Days"), "Potassh_50Days": hi("Potassh_50Days")}),
    ("T3", "Low fertilizer (min Urea and Potash)", {"Urea_40Days": lo("Urea_40Days"), "Potassh_50Days": lo("Potassh_50Days")}),
    ("T4", "Variety = ponmani", {"Variety": "ponmani"}),
    ("T5", "Variety = delux ponni", {"Variety": "delux ponni"}),
    ("T6", "Soil = clay", {"Soil Types": "clay"}),
    ("T7", "Nursery = wet", {"Nursery": "wet"}),
    ("T8", "Agriblock = Panruti", {"Agriblock": "Panruti"}),
    ("T9", "Small farm (min hectares)", {"Hectares": lo("Hectares")}),
    ("T10", "Large farm (max hectares)", {"Hectares": hi("Hectares")}),
    ("T11", "All inputs at minimum", {c: lo(c) for c in ranges}),
    ("T12", "All inputs at maximum", {c: hi(c) for c in ranges}),
    ("T13", "Edge case: unseen Variety value", {"Variety": "unknown_variety"}),
]

rows = []
for tid, desc, ch in tests:
    per_ha, total = predict(**ch)
    ok = np.isfinite(per_ha) and 2000 < per_ha < 10000      # plausible paddy yield range
    rows.append({"Test": tid, "Description": desc,
                 "Predicted yield (kg/ha)": round(per_ha, 1),
                 "Total yield (kg)": round(total, 1),
                 "Result": "PASS" if ok else "CHECK"})
scen = pd.DataFrame(rows)
scen.to_csv(BASE / "scenario_test_results.csv", index=False)
print(scen.to_string(index=False))

# ---------- Part B: batch of real records ----------
df = pd.read_csv(DATA)
df.columns = [c.strip() for c in df.columns]
sample = df.sample(25, random_state=7)
actual = sample["Paddy yield(in Kg)"] / sample["Hectares"]
pred = model.predict(sample.drop(columns=["Paddy yield(in Kg)"])[cols])
batch = pd.DataFrame({"Actual (kg/ha)": actual.round(1).values, "Predicted (kg/ha)": pred.round(1),
                      "Error (kg/ha)": (pred - actual.values).round(1)})
batch["Abs error %"] = (batch["Error (kg/ha)"].abs() / batch["Actual (kg/ha)"] * 100).round(2)
batch.to_csv(BASE / "batch_test_results.csv", index=False)
print("\nBatch test on 25 real records:")
print("Mean absolute error: %.1f kg/ha | Mean abs error: %.2f %%" %
      (batch["Error (kg/ha)"].abs().mean(), batch["Abs error %"].mean()))
print("Note: these records were part of the final training data, so this shows the app pipeline "
      "works end to end; unseen-data accuracy is in 3_Performance_Evaluation_Metrics.")
