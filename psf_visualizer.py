# python 2 and 3 comptibility
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import matplotlib.pylab as plt
import numpy as np


class PSF_calculator:
    def __init__(self, difference_image, n_sigma=1.5, psf_iterations=5):

        self.n = n_sigma
        self.sigmas = np.empty((0))
        self.n_sigmas = np.empty((0))
        self.diff = difference_image

        self.diff_center = int(np.round(self.diff.shape[0] / 2))

        self.iterative_calculation_raw_PSF(psf_iterations)

    def get_n_sigma(self, image):
        sigma = np.std(image, axis=(0, 1))
        n_sigma = self.n * sigma
        self.sigmas = np.append(self.sigmas, sigma)
        self.n_sigmas = np.append(self.n_sigmas, n_sigma)
        return n_sigma

    def calculate_raw_PSF(self):
        n_sigma = self.get_n_sigma(self.raw_PSFs[-1])
        raw_PSF = self.diff * (self.diff > n_sigma)
        self.raw_PSFs.append(raw_PSF)

    def iterative_calculation_raw_PSF(self, iterations):
        self.sigmas = np.empty((0))
        self.n_sigmas = np.empty((0))
        self.raw_PSFs = [self.diff]
        for i in range(iterations):
            self.calculate_raw_PSF()

    def plot_psf_sigmas(self, idx=0, x_pos=31, y_pos=31):
        self.check_availability(self.n_sigmas, idx)

        x_pos = np.full(self.diff.shape[0], x_pos)
        y_pos = np.full(self.diff.shape[0], y_pos)
        line = np.arange(0, self.diff.shape[0])

        x_axis = self.diff[:, y_pos[0]]
        y_axis = self.diff[x_pos[0], :]

        self.print_diff()
        plt.plot(line, x_pos, color='black', alpha=0.3)
        plt.plot(y_pos, line, color='black', alpha=0.5)
        plt.show()

        n_sigma_line = np.full(self.diff.shape[0], self.n_sigmas[idx])
        plt.bar(line, x_axis)
        plt.plot(line, n_sigma_line, color='black')
        plt.plot(line, -1 * n_sigma_line, color='black')
        plt.title(r'x_axis $\sigma$[' + str(idx) + r']: $\pm$' + str(self.n_sigmas[idx].round(decimals=3)))
        plt.show()

        plt.bar(line, y_axis)
        plt.plot(line, n_sigma_line, color='black')
        plt.plot(line, -1 * n_sigma_line, color='black')
        plt.title(r'y_axis $\sigma$[' + str(idx) + r']: $\pm$' + str(self.n_sigmas[idx].round(decimals=3)))
        plt.show()

        self.print_diff_raw_psf(idx=idx)
        plt.show()

        self.plot_array(self.n_sigmas)
        plt.title(r'$\sigma$´s progretions')
        plt.show()

    def plot_array(self, array):
        array = np.array(array)
        x = np.arange(0, array.shape[0])
        plt.plot(x, array)

    def print_sample(self, img, titles):
        if type(titles) != list:
            titles = [titles]
        fig = plt.figure()
        for k, imstr in enumerate(titles):
            ax = fig.add_subplot(1, len(titles), k + 1)
            ax.axis('off')
            ax.set_title(imstr)
            if len(titles) == 1:
                ax.matshow(img)
                return
            ax.matshow(img[k, ...])

    def print_diff(self):
        self.print_sample(self.diff, 'Difference')

    def print_raw_psf(self, idx=0):
        idx = idx + 1
        self.check_availability(self.raw_PSFs, idx)
        self.print_sample(self.raw_PSFs[idx], 'raw PSF ' + str(idx - 1))

    def print_diff_raw_psf(self, idx=0):
        idx = idx + 1
        self.check_availability(self.raw_PSFs, idx)
        self.print_sample(np.array([self.diff[21:42, 21:42], self.raw_PSFs[idx][21:42, 21:42]]),
                          ['Difference', 'raw PSF ' + str(idx - 1)])

    def check_availability(self, array, idx):
        array = np.array(array)
        if array.shape[0] - 1 < idx:
            self.iterative_calculation_raw_PSF(idx + 1)
