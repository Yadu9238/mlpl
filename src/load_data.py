import pandas as pd
import os
import argparse
import yaml 
from get_data import read_data,read_params
from log import logger
logger = logger('logs/','preprocessing.log')
def load_save_data(config_path):
    logger.info("In load_save_data")
    logger.info("Reading config")
    config = read_params(config_path)
    logger.info("Received configs")
    logger.info("Reading data with details from config")
    data = read_data(config_path)
    logger.info("Finished reading data")
    cols = [c.split('(')[0] for c in data.columns]

    raw_data_path = config['load_data']['raw_data']
    logger.info("Saving data locally in path: "+str(raw_data_path))
    data.to_csv(raw_data_path,sep=',',index = False,header = cols)
    


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--config",default="params.yaml")
    parsed_args = args.parse_args()
    load_save_data(parsed_args.config)
