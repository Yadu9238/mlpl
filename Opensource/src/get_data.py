import pandas as pd
import argparse
import yaml
from log import logger

logger = logger('logs/', 'preprocessing.log')


def read_params(config_path):
    logger.info("In read_params")
    logger.info('Trying to read '+str(config_path))
    with open(config_path) as f:
        config = yaml.safe_load(f)
    logger.info("Parameters read")
    logger.info("Returning configs")
    return config


def read_data(config_path):
    logger.info("In read_data")
    config = read_params(config_path)
    # print(config)
    data_path = config['data_source']['raw_data']
    logger.info("Provided data path is: "+str(data_path))
    print(data_path)
    data = pd.read_csv(data_path, sep=',', encoding='utf-8')
    logger.info("Data read from the provided file")
    return data


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()
    data = read_data(parsed_args.config)
    # print(data)
