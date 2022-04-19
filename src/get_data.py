import pandas as pd
import os
import argparse
import yaml 

def read_params(config_path):
    with open(config_path) as f:
        config = yaml.safe_load(f)

    return config

def read_data(config_path):
    config = read_params(config_path)
    #print(config)
    data_path = config['data_source']['raw_data']
    print(data_path)
    data = pd.read_csv(data_path,sep =',',encoding='utf-8')
    print(data.head())

if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--config",default="params.yaml")
    parsed_args = args.parse_args()
    read_data(parsed_args.config)
