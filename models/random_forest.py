from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
def train_model(X_train,y_train,n_estimators=100,max_depth=15,min_samples_leaf=5):

    model = RandomForestRegressor(
    n_estimators= n_estimators,
    max_depth= max_depth,
    min_samples_leaf=min_samples_leaf,
    random_state=42,
    n_jobs=-1)
    model.fit(X_train,y_train)
    return model
def evaluate_model(model,X_validation,y_validation):
    y_validation_pred = model.predict(X_validation)
    mae = mean_absolute_error(y_validation, y_validation_pred)
    rmse = np.sqrt(mean_squared_error(y_validation, y_validation_pred))
    r2 = r2_score(y_validation, y_validation_pred)
    return y_validation_pred, mae, rmse, r2
