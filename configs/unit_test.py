from config import config
from merge_config import merge_config
import argparse
import importlib

parser = argparse.ArgumentParser(description='Train image model with cross entropy loss')
# Datasets
parser.add_argument('--root', type=str, default='data',
                    help="root path to data directory")
parser.add_argument('-dataset', '--dataset', type=str, default='market1501')

args = parser.parse_args()

if __name__ == '__main__':

    CONFIG = importlib.import_module('resnet50_bagOfTricks')
    config = merge_config(CONFIG.config, args)
    print(config)