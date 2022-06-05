
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import numpy as np
import mlflow
from log import logger
from azureml.core.run import Run

logger = logger('logs/', 'model.log')
run = Run.get_context()
ws = run.experiment.workspace
print("workspace = ",ws)
mlflow.set_tracking_uri(ws.get_mlflow_tracking_uri())
exp_name = 'Cement Strength Prediction'
mlflow.set_experiment(exp_name)


def eval(y_pred, y_test):
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return rmse, mae, r2


def build_model(X_train, X_test, y_train, y_test):
    logger.info("In build_model")
    model = GradientBoostingRegressor()
    model.fit(X_train,y_train)
    y_pred = model.predict(X_test)
    rmse, mae, r2 = eval(y_pred, y_test)
    logger.info("Root mean square error: "+str(rmse))
    logger.info("Mean absolute error: "+str(mae))
    logger.info("R2 score: "+str(r2))
    logger.info("################################")
    mlflow.log_metric('RMSE', rmse)
    mlflow.log_metric('MAE', mae)
    mlflow.sklearn.log_model(model, "model")
    return model
    