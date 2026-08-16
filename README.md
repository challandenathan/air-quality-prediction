# air-quality-prediction

This project predicts PM2.5 concentrations using historical weather observations and machine learning models.
We are aiming to create weather-only model and also a weather + pollution model
## Dataset
The project uses historical air-quality measurements from monitoring stations in Beijing. (Beijing Multi-Site Air Quality Dataset (UCI)).

The dataset contains measurements including:
Meteorological variables:
TEMP — temperature
PRES — atmospheric pressure
DEWP — dew point temperature
RAIN — rainfall
WSPM — wind speed
wd — wind direction

Air-pollution variables:
PM2.5 — target variable
PM10
SO2
NO2
CO
O3

## Data preprocessing
- Missing target value: 
Rows with missing PM2.5 values are removed because they cannot be used as supervised training examples.
Meteorological variables

- Missing meteorological values are interpolated separately for each monitoring station.
The following variables are interpolated:
TEMP
PRES
DEWP
RAIN
WSPM

- Wind direction
Wind direction is treated as a categorical variable.
Missing values are filled using the most frequent wind direction for the corresponding station, calculated from the training data.

- Temporal features
The original date information is transformed into cyclical features.
For example, the hour of the day is represented using: hour_sin & hour_cos
and similarly for:
month_sin
month_cos
day_of_week_sin
day_of_week_cos
This allows the model to capture the periodic nature of temporal variables.

- Categorical features
The following variables are one-hot encoded: wd & stations.
The encoder is fitted on the training data and saved for use during live inference.

## Train / validation / test split

Because air-quality measurements are temporal data, the dataset is split chronologically rather than randomly.
Training:
before 2016-01-01
Validation:
2016-01-01 → 2016-07-01
Test:
from 2016-07-01

## Models

- Linear Regression
- Random Forest
- XGBoost
For each architectures, we use 2 differents features set: 
- model 1 use only meteorological features :
TEMP
PRES
DEWP
RAIN
WSPM
hour_sin
hour_cos
month_sin
month_cos
day_of_week_sin
day_of_week_cos
wd
station
- model 2 use the same features + pollution features:
PM10
SO2
NO2
CO
O3

For the evaluation of the models performances we use Mean Absolute Error (MAE), Root Mean Squared Error (RMSE) AND R2.

For the final model, XGBoost has been selected. The model use the following parameters:
n_estimators=1544 (defined by early stopping technique)
learning_rate=0.10
max_depth=3
min_child_weight=5
subsample=1.0
colsample_bytree=1.0
The final trained model is saved in: models/saved/xgboost_pollution.json

The corresponding one-hot encoder is saved in:
models/saved/one_hot_encoder.joblib

This ensures that the exact preprocessing used during training can also be reproduced during inference.

## API

The API used to gather the current data is Open-Meteo.
The trained model is exposed through a REST API using FastAPI.
The API currently provides endpoints for:
GET /
GET /stations
GET /predict?station=<station_name>
The API currently supports the 12 Beijing monitoring stations:
Aotizhongxin
Changping
Dingling
Dongsi
Guanyuan
Gucheng
Huairou
Nongzhanguan
Shunyi
Tiantan
Wanliu
Wanshouxigong

Each station is associated with latitude and longitude coordinates.

## Project structure
air-quality-prediction/
│
├── app/
│   └── main.py
│
├── src/
│   ├── api.py
│   ├── preprocessing.py
│   └── load_data.py
│   └──__init__.py
│
├── models/
│   ├── linear_regression.py
│   ├── random_forest.py
│   ├── xgboost.py
│   └── saved/
│       ├── xgboost_pollution.json
│       └── one_hot_encoder.joblib
│
├── notebooks/
│   └── model_analysis.ipynb
│   └── test_api.ipynb
│   └──exploration.ipynb
│   └──xgboost_grid_search_results.csv
│ 
├── data/
│   └── raw/
│       └── *.csv
│   └──processed/
│
│
├── environment.yml
├── .gitignore
└── README.md

## Installation
- Clone the repository:
git clone <https://github.com/challandenathan/air-quality-prediction>
cd air-quality-prediction
- Create the Conda environment:
conda env create -f environment.yml
- Activate it: 
conda activate air-quality

## Running the API
Start the FastAPI development server:
fastapi dev
The API will be available at:
http://127.0.0.1:8000
FastAPI also provides an interactive API documentation interface at:
http://127.0.0.1:8000/docs
From there, the available endpoints can be tested directly from the browser.
Example:
To predict the current PM2.5 concentration at Aotizhongxin:
GET /predict?station=Aotizhongxin
The API retrieves the latest available environmental information for the station and feeds the resulting feature vector into the trained XGBoost model.

## Streamlit Frontend
A Streamlit frontend provides a simple interactive interface for the live prediction API.

The frontend communicates with the FastAPI backend rather than running the machine-learning model directly. Users can select one of the available Beijing monitoring stations and request a live PM2.5 prediction.

The dashboard displays:

- Selected monitoring station
- Station coordinates
- Predicted PM2.5 concentration
- Current meteorological conditions
- Current pollutant concentrations
- Data timestamp
- Open-Meteo PM2.5 reference value, when available

To start the frontend, first launch the FastAPI backend: fastapi dev

Then, in a second terminal: streamlit run frontend/app.py


## Technologies
The project uses:

- Python
- Pandas
- NumPy
- scikit-learn
- XGBoost
- Joblib
- Requests
- FastAPI
- Uvicorn
- Open-Meteo API
- Jupyter Notebook
- Conda
- Streamlit


