import os
import pandas as pd
import numpy as np
import json
import argparse
import joblib
from model import build_model
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from get_data import read_params
from log import logger

logger = logger('logs/', 'training.log')


def eval(y_pred, y_test):
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    return rmse, mae, r2


def train(config_path):
    logger.info("In train")
    logger.info("Reading config file")
    config = read_params(config_path)
    train_data_path = config['split_data']['train_path']
    test_data_path = config['split_data']['test_path']

    target = config['base']['target_col']
    model_dir = config['model_dir']

    scores_file = config['report']['scores']
    params_file = config['report']['params']
    logger.info("Received config details:")
    logger.info("Train data path: "+str(train_data_path))
    logger.info("Test data path: "+str(test_data_path))
    logger.info("Target: "+str(target))
    logger.info("Model output: "+str(model_dir))
    logger.info("Evaluation metrics output: "+str(scores_file))
    logger.info("Best params after training: "+str(params_file))

    if not os.path.isfile(scores_file):
        open(scores_file, "w+").close()
    if not os.path.isfile(params_file):
        open(params_file, "w+").close()
    train_data = pd.read_csv(train_data_path, sep=',')
    test_data = pd.read_csv(test_data_path, sep=',')

    logger.info("Splitting data into dependent and independent variables")
    X_train = train_data.drop(target, axis=1)
    X_test = test_data.drop(target, axis=1)
    y_train = train_data[target]
    y_test = test_data[target]
    logger.info("Building model")
    model, params = build_model(X_train, X_test, y_train, y_test)
    logger.info("Received best fit model")
    model.fit(X_train, y_train)

    predicted = model.predict(X_test)
    (rmse, mae, r2) = eval(predicted, y_test)

    #print("RF model:(max_depth=%f,n_est=%f):"%(max_depth,n_estimators))
    print("RMSE = ", rmse)
    print("MAE = ", mae)
    print("R2 = ", r2)

    with open(scores_file, 'w+') as f:
        score = {
            "rmse": rmse,
            "mae": mae,
            "r2": r2
        }
        json.dump(score, f, indent=4)
    logger.info("Performance metrics of model saved in: "+str(scores_file))
    best_params = {}
    with open(params_file, 'w+') as f:
        for k, v in params.items():
            best_params[k] = v
        json.dump(best_params, f, indent=4)

    logger.info("Best parameters for model saved in: "+str(params_file))
    os.makedirs(model_dir, exist_ok=True)

    model_path = os.path.join(model_dir, "model.joblib")

    joblib.dump(model, model_path)
    logger.info("Model saved in: "+str(model_path))


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--config', default='params.yaml')
    parsed_args = args.parse_args()
    train(config_path=parsed_args.config)
