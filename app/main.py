from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field
import sys
from contextlib import asynccontextmanager
from pathlib import Path
#we connect the main file to the src folder:
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0,str(PROJECT_ROOT))
from src.api import (STATION_COORDINATES, predict_current_pm25,load_model_and_encoder)
@asynccontextmanager
async def lifespan(app: FastAPI):

    model, encoder = load_model_and_encoder()

    app.state.model = model
    app.state.encoder = encoder

    yield

class PredictionInputs(BaseModel):
    temperature: float = Field(
        description="Temperature at 2 m above ground (°C)"
    )
    pressure: float = Field(
        description="Surface pressure (hPa)"
    )
    dew_point: float = Field(
        description="Dew point temperature (°C)"
    )
    rain: float = Field(
        description="Rain (mm)"
    )
    wind_speed: float = Field(
        description="Wind speed at 10 m (m/s)"
    )
    wind_direction: str = Field(
        description="Wind direction"
    )
    pm10: float = Field(
        description="PM10 concentration (µg/m³)"
    )
    co: float = Field(
        description="Carbon monoxide concentration (µg/m³)"
    )
    no2: float = Field(
        description="Nitrogen dioxide concentration (µg/m³)"
    )
    so2: float = Field(
        description="Sulfur dioxide concentration (µg/m³)"
    )
    o3: float = Field(
        description="Ozone concentration (µg/m³)"
    )
class PredictionResponse(BaseModel):
    station: str
    timestamp: str
    latitude: float
    longitude: float
    inputs: PredictionInputs
    predicted_pm25: float = Field(
        description="Predicted PM2.5 concentration (µg/m³)"
    )
app = FastAPI(lifespan=lifespan)
@app.get("/")
def root():
    return{
        "message": "Air Quality Prediction API"
    }
@app.get("/stations")
def get_stations():
    return{
        "stations": list(STATION_COORDINATES.keys())
    }
@app.get("/health")
def health():
    return{"status": "ok"}
@app.get("/predict", response_model = PredictionResponse)
def predict(station: str,request: Request):

    if station not in STATION_COORDINATES:
        raise HTTPException(status_code = 404,
                            detail=f"Unknown station:{station}")

    result = predict_current_pm25(station,
                                  request.app.state.model,
                                  request.app.state.encoder)
    return PredictionResponse(
        station=station,
        timestamp=result["timestamp"],
        latitude=result["latitude"],
        longitude=result["longitude"],
        inputs=PredictionInputs(
            **result["inputs"]
        ),
        predicted_pm25=result["prediction"]
    )