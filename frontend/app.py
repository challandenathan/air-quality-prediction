#THE STREAMLIT HAS BEEN MADE WITH CHATGPT 

import streamlit as st
import requests


API_URL = "http://127.0.0.1:8000"


st.set_page_config(
    page_title="Beijing Air Quality",
    page_icon="🌍",
    layout="centered"
)


st.title("🌍 Beijing Air Quality Prediction")

st.write(
    "Select a Beijing monitoring station to obtain a "
    "live PM2.5 prediction."
)


# Get stations from FastAPI
try:

    response = requests.get(
        f"{API_URL}/stations",
        timeout=10
    )

    response.raise_for_status()

    stations = response.json()["stations"]

except requests.exceptions.RequestException as e:

    st.error(
        "Could not connect to the FastAPI server."
    )

    st.code(str(e))

    st.stop()


# Station selector
station = st.selectbox(
    "Select a monitoring station:",
    stations
)


st.write(f"Selected station: **{station}**")

# Prediction
if st.button("Predict PM2.5"):

    try:

        response = requests.get(
            f"{API_URL}/predict",
            params={"station": station},
            timeout=30
        )

        response.raise_for_status()

        result = response.json()

    except requests.exceptions.RequestException as e:

        st.error("Could not retrieve the prediction.")
        st.code(str(e))
        st.stop()

    # --------------------------------------------------
    # PM2.5 prediction
    # --------------------------------------------------

    prediction = result["predicted_pm25"]

    st.divider()

    st.subheader("PM2.5 Prediction")

    st.metric(
        label="Predicted PM2.5",
        value=f"{prediction:.2f} µg/m³"
    )

    # Current/reference PM2.5 is displayed separately.
    # We do not calculate an "error" because the timestamps
    # and data sources are not necessarily identical.
    if "measured_pm25" in result:

        st.metric(
            label="Open-Meteo PM2.5 reference",
            value=f"{result['measured_pm25']:.2f} µg/m³"
        )

        st.caption(
            "The Open-Meteo PM2.5 value is shown as a live reference. "
            "It is not necessarily a direct measurement from the "
            "selected Beijing monitoring station."
        )


    # --------------------------------------------------
    # Station information
    # --------------------------------------------------

    st.subheader("Station")

    col1, col2 = st.columns(2)

    with col1:
        st.write(f"**Station:** {result['station']}")

    with col2:
        st.write(
            f"**Coordinates:** "
            f"{result['latitude']:.4f}, "
            f"{result['longitude']:.4f}"
        )

    # --------------------------------------------------
    # Current meteorological conditions
    # --------------------------------------------------

    inputs = result["inputs"]

    st.subheader("Current conditions")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Temperature",
            f"{inputs['temperature']:.1f} °C"
        )

    with col2:
        st.metric(
            "Pressure",
            f"{inputs['pressure']:.1f} hPa"
        )

    with col3:
        st.metric(
            "Dew point",
            f"{inputs['dew_point']:.1f} °C"
        )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Rain",
            f"{inputs['rain']:.1f} mm"
        )

    with col2:
        st.metric(
            "Wind speed",
            f"{inputs['wind_speed']:.2f} m/s"
        )

    with col3:
        st.metric(
            "Wind direction",
            inputs["wind_direction"]
        )

    # --------------------------------------------------
    # Current pollutants
    # --------------------------------------------------

    st.subheader("Current pollutants")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "PM10",
            f"{inputs['pm10']:.1f} µg/m³"
        )

    with col2:
        st.metric(
            "NO₂",
            f"{inputs['no2']:.1f} µg/m³"
        )

    with col3:
        st.metric(
            "SO₂",
            f"{inputs['so2']:.1f} µg/m³"
        )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "O₃",
            f"{inputs['o3']:.1f} µg/m³"
        )

    with col2:
        st.metric(
            "CO",
            f"{inputs['co']:.1f} µg/m³"
        )

    # --------------------------------------------------
    # Timestamp
    # --------------------------------------------------

    st.divider()

    st.caption(
    f"Data timestamp: {result['timestamp']}"
    )