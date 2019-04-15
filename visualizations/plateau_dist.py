import numpy as np
import sys
import os
import matplotlib.pyplot as plt
import math
from mpl_toolkits.mplot3d import Axes3D

PATH_TO_PROJECT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(PATH_TO_PROJECT)


def plateau_dist_sin(x, a):
    k = (a / np.pi) * np.sin(np.pi / (2 * a))
    y = k * 1 / (1 + np.power(x, 2 * a))
    return y


def plateau_dist(x, beta, sigma, mu):
    k = beta / (2 * sigma * math.gamma(1 / beta))
    inside_exp = -np.power((np.abs(x - mu)**2 / sigma), beta)
    y = k * np.exp(inside_exp)
    return y


def plateau_dist_2d(x, y, beta, sigma, mu):
    k = beta / (2 * (sigma ** 2) * math.gamma(1 / beta))
    inside_exp = -np.power(((np.abs(x - mu)**2 + np.abs(y - mu)**2) / (sigma ** 2)), beta)
    z = k * np.exp(inside_exp)
    return z


if __name__ == "__main__":
    x = np.linspace(-5, 5, 100)
    y = np.linspace(-5, 5, 100)
    plt.plot(x, plateau_dist(x, beta=2, sigma=10, mu=0))
    plt.show()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    X, Y = np.meshgrid(x, y)
    zs = np.array(plateau_dist_2d(np.ravel(X), np.ravel(Y), beta=6, sigma=2, mu=0))
    Z = zs.reshape(X.shape)
    ax.plot_surface(X, Y, Z)
    plt.show()
