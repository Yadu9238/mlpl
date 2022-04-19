import pandas as pd
import os
import argparse
import yaml 
from get_data import read_data,read_params


def load_save_data(config_path):
    config = read_params(config_path)
    data = read_data(config_path)
    cols = [c.split('(')[0] for c in data.columns]

    raw_data_path = config['load_data']['raw_data']
    data.to_csv(raw_data_path,sep=',',index = False,header = cols)
    


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--config",default="params.yaml")
    parsed_args = args.parse_args()
    load_save_data(parsed_args.config)
