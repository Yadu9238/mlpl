import pandas as pd
import argparse
import yaml
from log import logger
from azureml.core import Workspace, Dataset
logger = logger('logs/', 'preprocessing.log')
from azureml.core import Dataset

from azureml.core.run import Run
run = Run.get_context()
workspace = run.experiment.workspace

def read_data(name):
    logger.info("In read_data")
    logger.info("checking for file name:{}".format(name))
    data = Dataset.get_by_name(workspace,name = name)
    logger.info("Data read from the provided file")
    return data.to_pandas_dataframe()


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--name", default="data")
    parsed_args = args.parse_args()
    data = read_data(parsed_args.name)
    # print(data)
