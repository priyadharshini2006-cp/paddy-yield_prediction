import sys, json, joblib, numpy as np, pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

out = sys.argv[1]
df = pd.read_csv(f"{out}/paddydataset.csv")
df.columns = [c.strip() for c in df.columns]
target = "Paddy yield(in Kg)"
# yield per hectare is the fair target (total yield scales with farm size)
df["yield_per_ha"] = df[target] / df["Hectares"]
X = df.drop(columns=[target, "yield_per_ha"])
y = df["yield_per_ha"]
cat = X.select_dtypes(exclude="number").columns.tolist()
num = [c for c in X.columns if c not in cat]
pre = ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), cat)], remainder="passthrough")
models = {
 "Ridge": Ridge(alpha=1.0),
 "RandomForest": RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1),
 "GradientBoosting": GradientBoostingRegressor(random_state=42),
}
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
res = {}
for n, m in models.items():
    p = Pipeline([("pre", pre), ("m", m)])
    cv = cross_val_score(p, Xtr, ytr, cv=KFold(5, shuffle=True, random_state=42), scoring="r2").mean()
    p.fit(Xtr, ytr); pr = p.predict(Xte)
    res[n] = dict(cv_r2=cv, MAE=mean_absolute_error(yte, pr), RMSE=mean_squared_error(yte, pr)**0.5, R2=r2_score(yte, pr))
    print(n, {k: round(v,3) for k,v in res[n].items()})
best = max(res, key=lambda k: res[k]["cv_r2"])
print("best:", best)
final = Pipeline([("pre", pre), ("m", models[best])]).fit(Xtr, ytr)
pr = final.predict(Xte)
print("test n", len(yte), "MAE", mean_absolute_error(yte, pr), "RMSE", mean_squared_error(yte, pr)**0.5, "R2", r2_score(yte, pr))
# refit on all data for the deployed model
final_all = Pipeline([("pre", pre), ("m", models[best])]).fit(X, y)
joblib.dump(final_all, f"{out}/yield_model.pkl")
meta = dict(best=best, results=res, columns=X.columns.tolist(), cat=cat,
  choices={c: sorted(X[c].unique().tolist()) for c in cat},
  ranges={c: [float(X[c].min()), float(X[c].max()), float(X[c].median())] for c in num},
  sklearn=__import__("sklearn").__version__)
json.dump(meta, open(f"{out}/model_meta.json","w"), indent=1)
imp = None
m = final_all.named_steps["m"]
if hasattr(m, "feature_importances_"):
    names = final_all.named_steps["pre"].get_feature_names_out()
    imp = sorted(zip(names, m.feature_importances_), key=lambda t:-t[1])[:10]
    print(imp)
