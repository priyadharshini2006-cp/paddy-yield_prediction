# 🌾 Paddy Yield Prediction

Machine learning web app that predicts paddy yield (kg/ha) from farm inputs, fertilizer and weather data.

**🔗 Live App:** https://paddy-yieldprediction-xqbhzptay9mbcbta4vnh3d.streamlit.app/

## Project Details
- Dataset: 2,789 farm records, 44 features (variety, soil, fertilizer, rainfall, temperature, humidity, etc.)
- Models compared: Ridge, Random Forest, Gradient Boosting
- Best model: Gradient Boosting (Test R² ≈ 0.50, MAE ≈ 155 kg/ha)
- Deployment: Streamlit Community Cloud

## Files
- `app.py` - Streamlit web app
- `train.py` - model training script
- `yield_model.pkl` - trained model
- `paddydataset.csv` - dataset

## Run locally
    pip install -r requirements.txt
    streamlit run app.py
