import os
import sys
import numpy as np
import pandas as pd
import pickle as pkl
import h5py
import re

PATH_TO_PROJECT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(PATH_TO_PROJECT)


if __name__ == "__main__":
  df_file_path = os.path.join(PATH_TO_PROJECT, '..', 'datasets', 'ZTF',
                              'alerts_corr_10detections.pickle')
  df_save_path = os.path.join(PATH_TO_PROJECT, '..', 'datasets', 'ZTF',
                              'alerts_corr_small.pickle')
  df = pd.read_pickle(df_file_path)
  df = df.iloc[:1000]
  df.to_pickle(df_save_path)