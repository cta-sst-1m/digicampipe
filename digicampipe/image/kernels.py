import numpy as np


def log(sigma=1., shape=None):  # Laplacien of Gaussian
    if shape is None:
        width = int(np.ceil(3 * sigma))
        shape = (width, width)
    result = np.zeros(shape)
    nbinx = shape[0]
    xmin = -(nbinx - 1) / 2
    x = np.arange(xmin, -xmin + .1, 1)
    nbiny = shape[1]
    ymin = -(nbiny - 1) / 2
    y = np.arange(ymin, -ymin + .1, 1)
    for binx in range(nbinx):
        for biny in range(nbiny):
            result[binx, biny] = (
                -(
                    1 - (x[binx] ** 2 + x[binx] ** 2) / (2 * sigma ** 2)
                ) *
                np.exp(
                    -(x[binx] ** 2 + y[biny] ** 2) / (2 * sigma ** 2)
                )
            )
    return result / np.sum(result)


def gauss(sigma=1., shape=None):
    if shape is None:
        width = int(np.ceil(3 * sigma))
        shape = (width, width)
    result = np.zeros(shape)
    nbinx = shape[0]
    xmin = -(nbinx - 1) / 2
    x = np.arange(xmin, -xmin + .1, 1)
    nbiny = shape[1]
    ymin = -(nbiny - 1) / 2
    y = np.arange(ymin, -ymin + .1, 1)
    for binx in range(nbinx):
        for biny in range(nbiny):
            result[binx, biny] = np.exp(
                (-x[binx] ** 2 - y[biny] ** 2) / (2 * sigma ** 2))
    return result / np.sum(result)


laplacien_33 = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
laplacien_55 = np.array([
    [-4, -1, 0, -1, -4],
    [-1, 2, 3, 2, -1],
    [0, 3, 4, 3, 0],
    [-1, 2, 3, 2, -1],
    [-4, -1, 0, -1, -4]])
sobelx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
sobely = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
sobelxy = 0.5 * (sobelx + sobely)
high_pass_filter_2525 = -np.ones((25, 25))/(25 * 25 - 1)
high_pass_filter_2525[12, 12] = 1
high_pass_filter_1313 = -np.ones((13, 13)) / (13 * 13 - 1)
high_pass_filter_1313[6, 6] = 1
high_pass_filter_77 = -np.ones((7, 7)) / (7 * 7 - 1)
high_pass_filter_77[3, 3] = 1
