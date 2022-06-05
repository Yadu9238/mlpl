import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from get_data import read_params
from log import logger

logger = logger('logs/', 'preprocessing.log')


def split_save_data(config_path):
    logger.info("In split_save_data")
    logger.info("Reading config file")
    config = read_params(config_path)
    logger.info("Received config details are:")
    train_data_path = config['split_data']['train_path']
    test_data_path = config['split_data']['test_path']
    raw_data_path = config['load_data']['raw_data']
    split_ratio = config['split_data']['test_size']
    random_state = config['base']['random_state']

    logger.info("Train data path: "+str(train_data_path))
    logger.info("Test data path: "+str(test_data_path))
    logger.info("Raw data path: "+str(raw_data_path))
    logger.info("Split ratio: "+str(split_ratio))
    logger.info("Random state: "+str(random_state))

    df = pd.read_csv(raw_data_path, sep=',')
    logger.info("Splitting data into Train and Test")
    train, test = train_test_split(
        df, test_size=split_ratio, random_state=random_state)
    logger.info("Saving train data into "+str(train_data_path))
    train.to_csv(train_data_path, sep=',', index=False)
    logger.info("Saving test data into "+str(test_data_path))
    test.to_csv(test_data_path, sep=',', index=False)


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()
    split_save_data(parsed_args.config)
