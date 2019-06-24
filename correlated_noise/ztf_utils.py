import numpy as np
import json
import base64
import io
import gzip
from astropy.io import fits
import matplotlib.pyplot as plt

def json2list(path):
    # load json
    with open(path, "r") as f:
        dataset = json.load(f)

    samples_list = []
    for i in range(len(dataset['query_result'])):

        channels = []
        for k, imstr in enumerate(['Template', 'Science', 'Difference']):
            stamp = dataset['query_result'][i]['cutout' + imstr]['stampData']
            stamp = base64.b64decode(stamp["$binary"].encode())

            with gzip.open(io.BytesIO(stamp), 'rb') as f:
                with fits.open(io.BytesIO(f.read())) as hdul:
                    img = hdul[0].data
                    channels.append(img)
        samples_list.append(np.array(channels))
    return samples_list


def createCircularMask(h, w, center=None, radius=None):
    if center is None:  # use the middle of the image
        center = [int(w / 2), int(h / 2)]
    if radius is None:  # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w - center[0], h - center[1])
    radius = np.round(radius)

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2)

    mask = dist_from_center <= radius
    return mask


def bin2array(binary):
    stamp = binary
    with gzip.open(io.BytesIO(stamp), 'rb') as f:
        with fits.open(io.BytesIO(f.read())) as hdul:
            img = hdul[0].data
    return img


def image_sigma_clip(stamp, n_iter=10, n_sigma=3, poisson_noise=False):
    aux_stamp = np.copy(stamp)
    current_av = np.mean(stamp)
    for i in range(n_iter):
        current_av = np.mean(aux_stamp)
        current_std = np.std(aux_stamp)
        large_values = (stamp > current_av+n_sigma*current_std)
        small_values = (stamp < current_av-n_sigma*current_std)
        down_shift = np.multiply(large_values, aux_stamp - current_av)
        up_shift = np.multiply(small_values, aux_stamp - current_av)
        aux_stamp = aux_stamp - down_shift
        aux_stamp = aux_stamp - up_shift
        if poisson_noise:
            aux_stamp[large_values] = np.random.poisson(np.abs(aux_stamp))[large_values]
            aux_stamp[small_values] = np.random.poisson(np.abs(aux_stamp))[small_values]
    return aux_stamp, current_av


def aperture_photometry(stamp, radius, center):
    h, w = stamp.shape
    mask = createCircularMask(h, w, radius=radius, center=center)
    masked = stamp[mask]
    counts = np.sum(masked)
    deviation = np.std(masked)
    masked_stamp = np.multiply(stamp, mask)
    return counts, deviation, masked_stamp


def sample_random_center(rows, cols, radius, n_centers = 10):
    random_rows = np.random.randint(low=radius, high=rows-radius, size=n_centers)
    random_cols = np.random.randint(low=radius, high=cols-radius, size=n_centers)
    valid_centers = np.stack([random_rows, random_cols], axis=1)
    return valid_centers


def run_background_photometry(background_images, appertures_per_stamp=10, radius=10, do_plots=False, return_masked=False):

    photometry_counts = []
    masked_stamps = []

    for i, stamp in enumerate(background_images):
        rows, cols = stamp.shape
        random_centers = sample_random_center(rows, cols,
                                              radius, n_centers=appertures_per_stamp)
        for j in range(random_centers.shape[0]):
            counts, dev, masked_stamp = aperture_photometry(stamp, radius, random_centers[j, ...])
            photometry_counts.append(counts)
            masked_stamps.append(masked_stamp)

        if do_plots:
            f, ax = plt.subplots(1, 2, figsize=(15, 7))
            im1 = ax[0].imshow(stamp)
            im2 = ax[1].imshow(masked_stamps[-1])
            ax[0].set_title("background_image")
            ax[1].set_title("masked")
            f.colorbar(im1, ax=ax[0], orientation='horizontal')
            f.colorbar(im2, ax=ax[1], orientation='horizontal')
            plt.show()

    if return_masked:
        return photometry_counts, masked_stamps
    else:
        return photometry_counts


















