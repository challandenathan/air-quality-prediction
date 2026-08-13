from sklearn.linear_model import LinearRegression
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def train_model(X_train,y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model
def evaluate_model(model, X_validation, y_validation):
    y_validation_pred = model.predict(X_validation)
    mae = mean_absolute_error(y_validation, y_validation_pred)
    rmse = np.sqrt(mean_squared_error(y_validation, y_validation_pred))
    r2 = r2_score(y_validation, y_validation_pred)
    return y_validation_pred, mae, rmse, r2

