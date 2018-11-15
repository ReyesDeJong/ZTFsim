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
%matplotlib inline

class data4psf_manager:
    def __init__(self, difference_image, n_sigma=1.5, psf_iterations=5):

        self.n = n_sigma
        self.sigmas = np.empty((0))
        self.n_sigmas = np.empty((0))
        self.diff = difference_image

        self.diff_center = int(np.round(self.diff.shape[0] / 2))

        self.iterative_calculation_raw_PSF(psf_iterations)