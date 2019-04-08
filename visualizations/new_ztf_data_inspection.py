import sys
import os
PATH_TO_PROJECT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(PATH_TO_PROJECT)
import numpy as np
import pickle as pkl
import pandas as pd


def load_pickle(path):
    infile = open(path, 'rb')
    dataset_partitions = pkl.load(infile)
    return dataset_partitions

if __name__ == "__main__":
    path_to_ztf_data = os.path.join(PATH_TO_PROJECT, '..', 'AlerceDHtest', 'datasets', 'ZTF')
    data_name = 'many_seen.pkl'
    data = load_pickle(os.path.join(path_to_ztf_data, data_name))
