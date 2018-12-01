#python 2 and 3 comptibility
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import json
import io
import gzip
import base64
from astropy.io import fits
import matplotlib.pylab as plt
import pickle as pkl
import numpy as np


class GetObservationConditions(object):

    def __init__(self, **kwargs):
        self.data_path = kwargs["data_path"]
        self.alerts_dataset = self.load_json(self.data_path)


    def load_json(self, path):
        with open(path, "r") as f:
            dataset = json.load(f)
        return dataset

    def build_field_data(self, keys_per_field):
        n_alerts = {}
        for alert in self.alerts_dataset["query_result"]:
            field = alert["candidate"]["field"]
            stamp = alert['cutoutDifference']['stampData']
            stamp = base64.b64decode(stamp["$binary"].encode())
            with gzip.open(io.BytesIO(stamp), 'rb') as f:
                with fits.open(io.BytesIO(f.read())) as hdul:
                    img = hdul[0].data
            jd = alert["candidate"]["jd"]
            if fwhm in n_alerts.keys():
                n_alerts[fwhm] += 1
                stamp_dict[fwhm].append(img)
                FWHM_dict[fwhm].append(jd)
            else:
                n_alerts[fwhm] = 1
                stamp_dict[fwhm] = [img, ]
                FWHM_dict[fwhm] = [jd, ]

if __name__ == "__main__":

    data_path = '/home/ereyes/Alerce/ZTF_7_18/broker_reals.json'
    image_path = './images/'
    keys_per_field = ['']
    alerts = GetObservationConditions(data_path=data_path)