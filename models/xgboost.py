import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
def train_model(X_train,y_train,n_estimators=500,learning_rate=0.05,max_depth=6,subsample=0.8,colsample_bytree=0.8):
    model = XGBRegressor(
    n_estimators=n_estimators,
    learning_rate=learning_rate,
    max_depth=max_depth,
    subsample=subsample,
    colsample_bytree=colsample_bytree,
    objective="reg:squarederror",
    random_state=42,
    n_jobs=-1
)
    model.fit(X_train,y_train)
    return model
def evaluate_model(model,X_validation,y_validation):
    y_validation_pred = model.predict(X_validation)
    mae = mean_absolute_error(y_validation, y_validation_pred)
    rmse = np.sqrt(mean_squared_error(y_validation, y_validation_pred))
    r2 = r2_score(y_validation, y_validation_pred)
    return y_validation_pred, mae, rmse, r2