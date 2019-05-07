import os
import sys
import numpy as np
import pandas as pd

PATH_TO_PROJECT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(PATH_TO_PROJECT)

class LightCurveDataframe2Array(object):

  def __init__(self, object_id_key = 'oid', filter_id_key = 'fid'):
    self.object_id_key = object_id_key
    self.filter_id_key = filter_id_key





if __name__ == "__main__":
  file_path = os.path.join(PATH_TO_PROJECT, '..', 'datasets', 'ZTF',
                           'alerts_corr_10detections.pickle')
  df = pd.read_pickle(file_path)
  print(df.keys())
  print(df[df['ndet']==32])
