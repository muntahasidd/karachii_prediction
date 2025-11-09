# ========================================
# 🌤️ Karachi AQI Forecast Streamlit App
# ========================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import requests
from datetime import datetime, timedelta
import plotly.express as px
import os

# ================= CONFIG =================
API_KEY = os.getenv("OPENWEATHER_API_KEY")

LAT, LON = 24.8607, 67.0011
CSV_PATH = "karachi_weather_5h.csv"
MODEL_PATH = "rf_model.pkl"
SCALER_PATH = "scaler.pkl"
SCALER_COLS_PATH = "scaler_columns.pkl"
LAG = 15
FORECAST_STEPS = 9  # ~3 days (5-hour intervals)

# ================= HELPER FUNCTIONS =================
def fetch_latest_weather():
    url = "http://api.openweathermap.org/data/2.5/weather"
    params = {"lat": LAT, "lon": LON, "appid": API_KEY}
    r = requests.get(url, params=params)
    data = r.json()
    return {
        "temperature": round(data["main"]["temp"] - 273.15, 2),
        "humidity": data["main"]["humidity"],
        "wind_speed": data["wind"]["speed"]
    }

def fetch_latest_pollution():
    url = "http://api.openweathermap.org/data/2.5/air_pollution"
    params = {"lat": LAT, "lon": LON, "appid": API_KEY}
    r = requests.get(url, params=params)
    data = r.json()["list"][0]
    comp = data["components"]
    return {
        "aqi": data["main"]["aqi"],
        "pm2_5": comp["pm2_5"],
        "pm10": comp["pm10"]
    }

def create_lag_features(df, lag=LAG):
    features = {}
    cols_to_lag = ["aqi", "pm2_5", "pm10", "temperature", "humidity", "wind_speed"]
    for col in cols_to_lag:
        for i in range(1, lag + 1):
            features[f"{col}_lag_{i}"] = df[col].iloc[-i]
    return pd.DataFrame([features])

def aqi_color(aqi):
    """Return color for decimal AQI (0-5 scale)"""
    if aqi < 1:      return "green"
    elif aqi < 2:    return "yellow"
    elif aqi < 3:    return "orange"
    elif aqi < 4:    return "red"
    elif aqi < 5:    return "purple"
    else:            return "maroon"

# ================= LOAD MODEL & SCALER =================
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
scaler_columns = joblib.load(SCALER_COLS_PATH)

# ================= STREAMLIT UI =================
st.set_page_config(page_title="Karachi AQI Forecast", layout="wide")
st.title("🌫️ Karachi AQI Forecast System")
st.markdown("Predict AQI for the **next 3 days** (5-hour intervals) using a trained Random Forest model.")

if st.button("📡 Fetch Latest Data & Predict"):
    try:
        df = pd.read_csv(CSV_PATH)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    except FileNotFoundError:
        st.error(f"❌ The CSV file `{CSV_PATH}` was not found! Please make sure it’s in the same directory.")
        st.stop()

    # ✅ Fetch latest real-time data
    latest_weather = fetch_latest_weather()
    latest_pollution = fetch_latest_pollution()
    latest_record = {**latest_weather, **latest_pollution, "timestamp": datetime.now()}
    df = pd.concat([df, pd.DataFrame([latest_record])], ignore_index=True)

    st.subheader("📊 Latest Weather & Air Quality Data")
    st.json(latest_record)

    # ✅ Create lag features
    X_input = create_lag_features(df.tail(LAG + 1), lag=LAG)
    X_input = X_input.reindex(columns=scaler_columns, fill_value=0)

    # ✅ Scale data
    X_scaled = scaler.transform(X_input)

    # ✅ Predict AQI
    forecast_scaled = model.predict(X_scaled).flatten()  # decimal values 0-5

    # ✅ Split into 3-day forecast
    day1, day2, day3 = np.array_split(forecast_scaled, 3)

    # ================= DISPLAY 3-DAY AVG =================
    st.subheader("🌟 3-Day Average AQI")
    for i, day in enumerate([day1, day2, day3], start=1):
        avg_aqi = np.mean(day)
        st.metric(f"Day {i} Avg AQI", f"{avg_aqi:.2f}")
        st.markdown(f"Category: **{aqi_color(avg_aqi).capitalize()}**")

    # ================= PLOT INTERACTIVE CHART =================
    st.subheader("📈 AQI Forecast Over Next 3 Days (5-hour intervals)")
    forecast_timestamps = pd.date_range(start=datetime.now(), periods=len(forecast_scaled), freq='5H')
    forecast_df = pd.DataFrame({
        "Timestamp": forecast_timestamps,
        "AQI": forecast_scaled
    })
    forecast_df["Color"] = [aqi_color(a) for a in forecast_df["AQI"]]

    fig = px.bar(forecast_df, x="Timestamp", y="AQI", color="Color",
                 title="Karachi AQI Forecast (0-5 scale, decimals)",
                 color_discrete_map={
                     "green":"green", "yellow":"yellow", "orange":"orange",
                     "red":"red", "purple":"purple", "maroon":"maroon"
                 })
    st.plotly_chart(fig, use_container_width=True)

    # ================= SHOW FULL FORECAST =================
    st.subheader("📋 Full Forecast (5-hour intervals)")
    forecast_df_display = forecast_df.copy()
    forecast_df_display["AQI"] = forecast_df_display["AQI"].round(2)
    st.dataframe(forecast_df_display)

st.markdown("---")
