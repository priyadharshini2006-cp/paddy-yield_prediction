# Model Deployment

Contains the exact files used by the deployed application.

| File | Purpose |
|---|---|
| `app.py` | Streamlit web app (user inputs, model loading, prediction display) |
| `yield_model.pkl` | Trained Gradient Boosting pipeline (one-hot encoding + model) |
| `model_meta.json` | Feature names, input ranges and category lists used to build the input form |
| `requirements.txt` | Python libraries and versions needed by the app |

## How the app works
1. User selects field details (Agriblock, Variety, Soil, Nursery) and enters farm inputs (hectares, fertilizer, seed rate, etc.). Weather values default to dataset averages.
2. On **Predict yield**, the inputs are converted to a one-row table in the same column order as training.
3. The saved pipeline predicts **yield per hectare (kg/ha)**.
4. The app shows kg/ha, total yield (kg/ha x hectares) and the typical model error (about 155 kg/ha).

## Deployment steps
1. Save the trained model: `joblib.dump(model, "yield_model.pkl")`
2. Write `app.py` and `requirements.txt` (pin the same scikit-learn version used for training: 1.9.1)
3. Test locally: `python -m streamlit run app.py` (opens at `localhost:8501`)
4. Push the files to the root of a public GitHub repository
5. On share.streamlit.io: **Create app** -> choose repository, branch `main`, main file `app.py`, Python 3.12 -> **Deploy**
6. Streamlit builds the environment from `requirements.txt` and publishes the public link

## Run locally
```
pip install -r requirements.txt
python -m streamlit run app.py
```
Note: the deployed copy of these files sits in the repository root (Streamlit needs `app.py` there). This folder is the same code for documentation.
