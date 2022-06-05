import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from get_data import read_data
from log import logger

logger = logger('logs/', 'preprocessing.log')


def split_save_data(dataFrame):
    logger.info("In split_save_data")
    logger.info("Received config details are:")
    
    logger.info("Splitting data into Train and Test")
    train, test = train_test_split(
        dataFrame, test_size=0.2, random_state=37)
    logger.info('Data split into train and test')
    return train,test


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--name", default="data")
    parsed_args = args.parse_args()
    Dataframe = read_data(parsed_args.name)
    train,test = split_save_data(Dataframe)
    print("Train size is :",train.shape)
    print("Test size is:",test.shape)
