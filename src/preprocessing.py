import pandas as pd
from pathlib import Path
import sys
import numpy as np
from sklearn.preprocessing import OneHotEncoder
#load the raw data for preprocessing
sys.path.append(str(Path("..").resolve()))
from src.load_data import load_data
df = load_data("data/raw")

df["datetime"] = pd.to_datetime(df[["year", "month", "day", "hour"]])
df = df.sort_values("datetime").reset_index(drop=True)
#missingness by station
missing_by_station = df.groupby("station").apply(lambda x: x.isna().mean() * 100)
missing_by_station.to_csv("missingness_by_station.csv", index=True)
#missingness by variable
missing_summary = pd.DataFrame({"missing_count": df.isna().sum(), "missing_percentage": df.isna().mean() * 100})
missing_summary.sort_values(
    "missing_percentage",
    ascending=False
)
missing_summary.to_csv("missingness_summary.csv", index=True)

#for a predictive task it is not possible to keep the data vectors without the target value, therefor we drop all of them
#=> going to remove 2.08 % of the data set
def remove_missing_target(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna(subset=["PM2.5"]).copy()
    return df

df_clean = remove_missing_target(df)
#then we sort the data for the train-validation-test split
df_clean = df_clean.sort_values("datetime").reset_index(drop=True)
train = df_clean[df_clean["datetime"]< "2016-01-01"].copy()

validation = df_clean[
    (df_clean["datetime"] >="2016-01-01") &
    (df_clean["datetime"] < "2016-07-01")
].copy()

test = df_clean[
    df_clean["datetime"] >= "2016-07-01"
].copy()
#interpolation of the weather features on the training dataset
weather_columns = [
    "TEMP",
    "PRES",
    "DEWP",
    "RAIN",
    "WSPM"
]

train = train.sort_values(
    ["station", "datetime"]
).copy()
train[weather_columns] = (train.groupby("station")[weather_columns].transform(lambda x: x.interpolate()))
#now we interpolate for the validation and test datset
validation = validation.sort_values(["station","datetime"]).copy()
test = test.sort_values(["station","datetime"]).copy()
validation[weather_columns] = (validation.groupby("station")[weather_columns].transform(lambda x: x.interpolate()))
test[weather_columns] = (test.groupby("station")[weather_columns].transform(lambda x: x.interpolate()))

#now we preprocess the wind direction data which is a categorical feature

#we fill the missing values with the most frequent value in the training dataset for each station
wd_mode = train.groupby("station")["wd"].agg(lambda x: x.mode().iloc[0])
train["wd"] = train["wd"].fillna(train["station"].map(wd_mode))
validation["wd"] = validation["wd"].fillna(validation["station"].map(wd_mode))
test["wd"] = test["wd"].fillna(test["station"].map(wd_mode))

#we finished the preprocessing of the weather variables, we now start the preprocessing for the pollution variables (useful for model 2)
pollution_columns = [
    "PM10",
    "SO2",
    "NO2",
    "CO",
    "O3"
]

#now we create the temporal features for both models
#we will create hours, days of the week and months features with periodicity encoding using sine and cosine
def add_cyclic_features(df):
    df = df.copy()

    df["hour"] = df["datetime"].dt.hour
    df["month"] = df["datetime"].dt.month
    df["day_of_week"] = df["datetime"].dt.dayofweek

    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    df["day_of_week_sin"] = np.sin(
        2 * np.pi * df["day_of_week"] / 7
    )
    df["day_of_week_cos"] = np.cos(
        2 * np.pi * df["day_of_week"] / 7
    )

    return df
train = add_cyclic_features(train)
validation = add_cyclic_features(validation)
test = add_cyclic_features(test)
train_pollution = train.dropna(subset= pollution_columns).copy()
validation_pollution = validation.dropna(
    subset=pollution_columns
).copy()
test_pollution = test.dropna(subset= pollution_columns).copy()

#for the encoding of the wind direction features we will use one hot encoding rather than digits
categorical_features = ["wd","station"]
encoder = OneHotEncoder(handle_unknown="ignore",sparse_output=False)
encoder.fit(train[categorical_features])
X_train_cat = encoder.transform(train[categorical_features])
X_validation_cat = encoder.transform(validation[categorical_features])
X_test_cat = encoder.transform(test[categorical_features])
X_train_pollution_cat = encoder.transform(
    train_pollution[categorical_features]
)
X_validation_pollution_cat = encoder.transform(
    validation_pollution[categorical_features]
)
X_test_pollution_cat = encoder.transform(
    test_pollution[categorical_features]
)
#now we create the preprocessed datasets for both models
#for model 1:
numerical_features = [
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
    "day_of_week_cos"
]
X_train_num = train[numerical_features].to_numpy()
X_validation_num = validation[numerical_features].to_numpy()
X_test_num = test[numerical_features].to_numpy()
#we concatenate the numerical and categorical features for the training, validation and test datasets
X_train = np.hstack([
    X_train_num,
    X_train_cat
])
X_validation = np.hstack([
    X_validation_num,
    X_validation_cat
])
X_test = np.hstack([
    X_test_num,
    X_test_cat
])
y_train = train["PM2.5"].to_numpy()
y_validation = validation["PM2.5"].to_numpy()
y_test = test["PM2.5"].to_numpy()
#for model 2:
numerical_features_pollution = [
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
X_train_pollution_num = train_pollution[
    numerical_features_pollution
].to_numpy()

X_validation_pollution_num = validation_pollution[
    numerical_features_pollution
].to_numpy()

X_test_pollution_num = test_pollution[
    numerical_features_pollution
].to_numpy()
X_train_pollution = np.hstack([
    X_train_pollution_num,
    X_train_pollution_cat
])

X_validation_pollution = np.hstack([
    X_validation_pollution_num,
    X_validation_pollution_cat
])

X_test_pollution = np.hstack([
    X_test_pollution_num,
    X_test_pollution_cat
])
y_train_pollution = train_pollution["PM2.5"].to_numpy()

y_validation_pollution = validation_pollution["PM2.5"].to_numpy()

y_test_pollution = test_pollution["PM2.5"].to_numpy()

#creation of the controlled dataset
#for model1
pollution_columns = [
    "PM10",
    "SO2",
    "NO2",
    "CO",
    "O3"
]
train_common = train.dropna(subset=pollution_columns).copy()
validation_common = validation.dropna(subset=pollution_columns).copy()
X_train_common_num = train_common[
    numerical_features
].to_numpy()

X_validation_common_num = validation_common[
    numerical_features
].to_numpy()
X_train_common_cat = encoder.transform(
    train_common[categorical_features]
)

X_validation_common_cat = encoder.transform(
    validation_common[categorical_features]
)
X_train_common = np.hstack([
    X_train_common_num,
    X_train_common_cat
])

X_validation_common = np.hstack([
    X_validation_common_num,
    X_validation_common_cat
])
y_train_common = train_common["PM2.5"].to_numpy()

y_validation_common = validation_common["PM2.5"].to_numpy()
#for model2
X_train_common_pollution_num = train_common[
    numerical_features_pollution
].to_numpy()

X_validation_common_pollution_num = validation_common[
    numerical_features_pollution
].to_numpy()
X_train_common_pollution_cat = encoder.transform(
    train_common[categorical_features]
)

X_validation_common_pollution_cat = encoder.transform(
    validation_common[categorical_features]
)
X_train_common_pollution = np.hstack([
    X_train_common_pollution_num,
    X_train_common_pollution_cat
])

X_validation_common_pollution = np.hstack([
    X_validation_common_pollution_num,
    X_validation_common_pollution_cat
])
y_train_common_pollution = train_common["PM2.5"].to_numpy()

y_validation_common_pollution = (
    validation_common["PM2.5"].to_numpy()
)