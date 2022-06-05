import os
import pandas as pd
import numpy as np
import json
import argparse
import joblib
from model import build_model
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from get_data import read_data
from log import logger
from split_data import split_save_data
from get_data import read_data
logger = logger('logs/', 'training.log')

from azureml.core.run import Run
run = Run.get_context()


def eval(y_pred, y_test):
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return rmse, mae, r2

def register_model(run, model_path):
    run.upload_file(model_path, "outputs/model.pkl")
    model = run.register_model(
        model_name='predictor',
        model_path="outputs/model.pkl"
    )
    logger.info("Model registered and uploaded with ID:",model.id)
    run.log('Model_ID', model.id)

def save_model(classifer):
    __here__ = os.path.dirname(__file__)
    output_dir = os.path.join(__here__, 'outputs')
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, 'model.pkl')
    joblib.dump(classifer, model_path)
    return model_path

def train(name):
    logger.info("In train")
    df = read_data(name)
    target = 'Concrete compressive strength'
    
    logger.info('Target column is {}'.format(target))
    logger.info("Received config details:")
    logger.info("Target: {}".format(target))
    
    train_data, test_data = split_save_data(df)

    logger.info("Splitting data into dependent and independent variables")
    
    X_train = train_data.drop(target, axis=1)
    X_test = test_data.drop(target, axis=1)
    y_train = train_data[target]
    y_test = test_data[target]
    
    logger.info("Building model")
    model = build_model(X_train, X_test, y_train, y_test)
    logger.info("Received best fit model")
    
    predicted = model.predict(X_test)
    (rmse, mae, r2) = eval(predicted, y_test)

    print("RMSE = ", rmse)
    print("MAE = ", mae)
    print("R2 = ", r2)
    model_path = save_model(model)
    register_model(run,model_path)
  
    


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--name', default='Cement dataset')
    parsed_args = args.parse_args()
    print(parsed_args.name)
    train(name=parsed_args.name)
