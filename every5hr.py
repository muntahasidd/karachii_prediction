# ========================================
# Karachi Weather & AQI Forecast - 5-hour Interval, Combined Historical + Recent Data
# ========================================

import requests, pandas as pd, numpy as np, time, os, sys
from datetime import datetime, timedelta
import hopsworks, joblib
from hsml.schema import Schema
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ======= CONFIG =======
os.environ["HOPSWORKS_API_KEY"] = os.getenv("HOPSWORKS_API_KEY")
API_KEY = os.getenv("OPENWEATHER_API_KEY")
LAT, LON = 24.8607, 67.0011
CITY_ID = 1174872
CSV_PATH = "karachi_weather_5h.csv"
FEATURE_STORE_NAME = "aqi_karachi_final"

# ======= FETCH FUNCTIONS =======
def fetch_weather_at(ts):
    url = "http://history.openweathermap.org/data/2.5/history/city"
    params = {
        "id": CITY_ID,
        "type": "hour",
        "start": int(ts.timestamp()),
        "end": int((ts + timedelta(hours=1)).timestamp()),
        "appid": API_KEY
    }
    r = requests.get(url, params=params, timeout=30)
    return r.json()["list"][0] if r.status_code == 200 and r.json().get("list") else None

def fetch_pollution_at(ts):
    url = "https://api.openweathermap.org/data/2.5/air_pollution/history"
    params = {
        "lat": LAT,
        "lon": LON,
        "start": int(ts.timestamp()),
        "end": int((ts + timedelta(hours=1)).timestamp()),
        "appid": API_KEY
    }
    r = requests.get(url, params=params, timeout=30)
    return r.json()["list"][0] if r.status_code == 200 and r.json().get("list") else None

def build_record(ts):
    weather = fetch_weather_at(ts)
    pollution = fetch_pollution_at(ts)
    time.sleep(1)
    if not weather or not pollution:
        print(f"⚠️ Skipped {ts}: incomplete data")
        return None
    comp = pollution["components"]
    return {
        "timestamp": ts.strftime("%Y-%m-%d %H:%M:%S"),
        "aqi": pollution["main"]["aqi"],
        "pm2_5": comp.get("pm2_5"),
        "pm10": comp.get("pm10"),
        "temperature": round(weather["main"]["temp"] - 273.15, 2),
        "humidity": weather["main"]["humidity"],
        "wind_speed": weather["wind"]["speed"]
    }

# ======= COLLECT NEW DATA =======
def collect_data_5days_every5hours():
    start = datetime.now() - timedelta(days=5)
    end = datetime.now()
    current = start
    records = []

    while current < end:
        print(f"Fetching {current}")
        rec = build_record(current)
        if rec:
            records.append(rec)
        current += timedelta(hours=5)

    df_new = pd.DataFrame(records)
    print(f"✅ Collected {len(df_new)} new records.")
    return df_new

# ======= PREPROCESSING =======
def cap_and_scale(df, cols):
    df = df.copy()
    for col in cols:
        Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        IQR = Q3 - Q1
        df[col] = np.clip(df[col], Q1 - 1.5*IQR, Q3 + 1.5*IQR)
    scaler = MinMaxScaler()
    df[cols] = scaler.fit_transform(df[cols])
    return df, scaler

def prepare_multioutput_forecast_data(df, lag=5, horizon=5):
    if len(df) < lag + horizon:
        print(f"❌ Not enough rows. Need at least {lag + horizon}, got {len(df)}.")
        return None, None

    feature_cols = ["aqi", "pm2_5", "pm10", "temperature", "humidity", "wind_speed"]
    lag_features = []
    for col in feature_cols:
        lag_df = pd.concat([df[col].shift(i) for i in range(1, lag + 1)], axis=1)
        lag_df.columns = [f"{col}_lag_{i}" for i in range(1, lag + 1)]
        lag_features.append(lag_df)

    X = pd.concat(lag_features, axis=1)
    y = pd.concat([df["aqi"].shift(-i) for i in range(1, horizon + 1)], axis=1)
    y.columns = [f"target_t_plus_{i}" for i in range(1, horizon + 1)]

    final = pd.concat([X, y], axis=1).dropna()
    return final[X.columns], final[y.columns]

# ======= MAIN =======
if __name__ == "__main__":
    # 1️⃣ Collect new data
    df_new = collect_data_5days_every5hours()

    # 2️⃣ Connect to Hopsworks
    project = hopsworks.login(project=FEATURE_STORE_NAME)
    fs = project.get_feature_store()
    mr = project.get_model_registry()

    # 3️⃣ Read old data from Feature Store (if exists)
    try:
        fg = fs.get_feature_group("karachi_weather_5h", version=1)
        df_old = fg.read()
        print(f"✅ Read {len(df_old)} old records from Feature Store.")
    except Exception as e:
        print("⚠️ No existing Feature Group found, creating new one.")
        df_old = pd.DataFrame()

    # 4️⃣ Ensure timestamps are datetime
    if not df_old.empty:
        df_old["timestamp"] = pd.to_datetime(df_old["timestamp"])
    df_new["timestamp"] = pd.to_datetime(df_new["timestamp"])

    # 5️⃣ Combine old + new data
    df_combined = pd.concat([df_old, df_new], ignore_index=True)
    df_combined.drop_duplicates(subset=["timestamp"], inplace=True)
    df_combined.sort_values("timestamp", inplace=True)
    print(f"Total combined records: {len(df_combined)}")

    # 6️⃣ Insert updated data into Feature Store
    fg = fs.get_or_create_feature_group(
        name="karachi_weather_5h",
        version=1,
        description="5-hour interval weather and AQI for Karachi",
        primary_key=["timestamp"],
        event_time="timestamp"
    )
    fg.insert(df_combined)
    print("✅ Updated data inserted into Feature Store")

    # 7️⃣ Preprocess: outlier cap + scaling
    df_processed, scaler = cap_and_scale(df_combined, ["temperature", "humidity", "wind_speed", "pm2_5", "pm10", "aqi"])

    # 8️⃣ Prepare lagged features & targets
    X, y = prepare_multioutput_forecast_data(df_processed, lag=5, horizon=5)
    if X is None or y is None:
        sys.exit(1)

    # 9️⃣ Split train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    # 🔟 Train model
    rf_model = MultiOutputRegressor(RandomForestRegressor(random_state=42))
    rf_model.fit(X_train, y_train)

    # 1️⃣1️⃣ Evaluate
    test_pred = rf_model.predict(X_test)
    rf_mae = mean_absolute_error(y_test, test_pred)
    rf_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
    rf_r2 = r2_score(y_test, test_pred)
    rf_acc = max(0, (1 - rf_rmse)) * 100

    print("\n=== Testing Performance ===")
    print("MAE:", round(rf_mae, 4),
          "RMSE:", round(rf_rmse, 4),
          "R²:", round(rf_r2, 4),
          "Accuracy:", round(rf_acc, 2), "%")

    # 1️⃣2️⃣ Save model to Hopsworks
    joblib.dump(rf_model, "rf_model.pkl", compress=3)
    input_example = X.iloc[0]
    model_schema = Schema(X)

    model = mr.python.create_model(
        name="karachi_aqi_forecaster_5h",
        metrics={
            "accuracy": round(rf_acc, 2),
            "test_mae": round(rf_mae, 4),
            "test_rmse": round(rf_rmse, 4),
            "test_r2": round(rf_r2, 4)
        },
        model_schema=model_schema,
        input_example=input_example,
        description="Random Forest retrained with historical + new 5-day data"
    )
    saved_model = model.save("rf_model.pkl")
    print(f"✅ Model saved to Hopsworks | R²: {round(rf_r2, 4)} | Accuracy: {round(rf_acc, 2)}%")
