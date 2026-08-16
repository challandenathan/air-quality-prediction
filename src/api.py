import requests
import pandas as pd
import numpy as np
from pathlib import Path
from xgboost import XGBRegressor
import joblib
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "saved" / "xgboost_pollution.json"
ENCODER_PATH = PROJECT_ROOT / "models" / "saved" / "one_hot_encoder.joblib"
NUMERICAL_FEATURES_POLLUTION = [
    "TEMP",
    "PRES",
    "DEWP",
    "RAIN",
    "WSPM",
    "hour_sin",
    "hour_cos",
    "month_sin",
    "month_cos",
    "day_of_week_sin",
    "day_of_week_cos",
    "PM10",
    "SO2",
    "NO2",
    "CO",
    "O3"
]

CATEGORICAL_FEATURES = [
    "wd",
    "station"
]

def load_model_and_encoder():
    model = XGBRegressor()
    model.load_model(MODEL_PATH)
    encoder = joblib.load(ENCODER_PATH)
    return model, encoder



WEATHER_API_URL = "https://api.open-meteo.com/v1/forecast"
AIR_QUALITY_API_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"
def get_weather(latitude,longitude):
    params= {
        "latitude": latitude,
        "longitude": longitude,
        "current": [
            "temperature_2m",
            "surface_pressure",
            "dew_point_2m",
            "rain",
            "wind_speed_10m",
            "wind_direction_10m"
        ],
        "wind_speed_unit": "ms",
        "timezone": "auto"
    }
    response = requests.get(WEATHER_API_URL, params=params)
    response.raise_for_status()
    return response.json()
def get_air_quality(latitude, longitude):
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "current": [
            "pm10",
            "carbon_monoxide",
            "nitrogen_dioxide",
            "sulphur_dioxide",
            "ozone"
        ],
        "timezone": "auto"
    }

    response = requests.get(
        AIR_QUALITY_API_URL,
        params=params
    )

    response.raise_for_status()

    return response.json()
def degrees_to_wind_direction(degrees):
    directions = [
        "N",
        "NNE",
        "NE",
        "ENE",
        "E",
        "ESE",
        "SE",
        "SSE",
        "S",
        "SSW",
        "SW",
        "WSW",
        "W",
        "WNW",
        "NW",
        "NNW"
    ]
    index = int((degrees + 11.25) / 22.5) % 16

    return directions[index]

def create_api_features(weather, air_quality):
    weather_current = weather["current"]
    air_current = air_quality["current"]

    features = {
        "TEMP": weather_current["temperature_2m"],
        "PRES": weather_current["surface_pressure"],
        "DEWP": weather_current["dew_point_2m"],
        "RAIN": weather_current["rain"],
        "WSPM": weather_current["wind_speed_10m"],
        "wd": degrees_to_wind_direction(
            weather_current["wind_direction_10m"]
        ),
        "PM10": air_current["pm10"],
        "CO": air_current["carbon_monoxide"],
        "NO2": air_current["nitrogen_dioxide"],
        "SO2": air_current["sulphur_dioxide"],
        "O3": air_current["ozone"]
    }

    return features

def add_api_datetime_features(features, timestamp):
    dt = pd.to_datetime(timestamp)
    features["year"] = dt.year
    features["month"] = dt.month
    features["day"] = dt.day
    features["hour"] = dt.hour
    features["dayofweek"] = dt.dayofweek

    return features


def add_cyclic_features(df):
    df = df.copy()

    df["hour"] = df["datetime"].dt.hour
    df["month"] = df["datetime"].dt.month
    df["day_of_week"] = df["datetime"].dt.dayofweek

    df["hour_sin"] = np.sin(
        2 * np.pi * df["hour"] / 24
    )

    df["hour_cos"] = np.cos(
        2 * np.pi * df["hour"] / 24
    )

    df["month_sin"] = np.sin(
        2 * np.pi * df["month"] / 12
    )

    df["month_cos"] = np.cos(
        2 * np.pi * df["month"] / 12
    )

    df["day_of_week_sin"] = np.sin(
        2 * np.pi * df["day_of_week"] / 7
    )

    df["day_of_week_cos"] = np.cos(
        2 * np.pi * df["day_of_week"] / 7
    )

    return df
def create_api_dataframe(features, timestamp, station):
    features = features.copy()

    features["datetime"] = pd.to_datetime(timestamp)
    features["station"] = station

    return pd.DataFrame([features])


STATION_COORDINATES = {
    "Aotizhongxin": (39.982, 116.397),
    "Changping": (40.217, 116.230),
    "Dingling": (40.292, 116.220),
    "Dongsi": (39.929, 116.417),
    "Guanyuan": (39.929, 116.339),
    "Gucheng": (39.914, 116.184),
    "Huairou": (40.328, 116.628),
    "Nongzhanguan": (39.937, 116.461),
    "Shunyi": (40.127, 116.655),
    "Tiantan": (39.886, 116.407),
    "Wanliu": (39.987, 116.287),
    "Wanshouxigong": (39.878, 116.352),
}

def predict_current_pm25(station,model,encoder):
    if station not in STATION_COORDINATES:
        raise ValueError(f"Unknown station: {station}")
    latitude, longitude = STATION_COORDINATES[station]
    
    weather = get_weather(latitude, longitude)
    air_quality = get_air_quality(latitude, longitude)

    
    features = create_api_features(
        weather,
        air_quality
    )

    timestamp = air_quality["current"]["time"]

    api_df = create_api_dataframe(
        features,
        timestamp,
        station
    )


    api_df = add_cyclic_features(api_df)

    api_numerical = api_df[
        NUMERICAL_FEATURES_POLLUTION
    ]

    api_categorical = encoder.transform(
        api_df[CATEGORICAL_FEATURES]
    )

    api_categorical_df = pd.DataFrame(
        api_categorical,
        columns=encoder.get_feature_names_out(
            CATEGORICAL_FEATURES
        )
    )


    X_api = pd.concat(
        [
            api_numerical.reset_index(drop=True),
            api_categorical_df.reset_index(drop=True)
        ],
        axis=1
    )


    X_api = X_api.to_numpy()

    prediction = float(model.predict(X_api)[0])

    prediction = max(0.0, prediction)

    return {"prediction":round(prediction,2),
            "timestamp": timestamp,
            "latitude": latitude,
            "longitude": longitude,
            "inputs": {
        "temperature": features["TEMP"],
        "pressure": features["PRES"],
        "dew_point": features["DEWP"],
        "rain": features["RAIN"],
        "wind_speed": features["WSPM"],
        "wind_direction": features["wd"],
        "pm10": features["PM10"],
        "co": features["CO"],
        "no2": features["NO2"],
        "so2": features["SO2"],
        "o3": features["O3"]
    }}