import numpy as np


def gaussian(x, mean, sigma, amplitude):

    pdf = (x - mean)**2 / (2 * sigma**2)
    pdf = np.exp(-pdf)
    pdf /= np.sqrt(2 * np.pi) * sigma
    pdf *= amplitude

    return pdf


def single_photoelectron_pdf(x, baseline, gain,
                             sigma_e, sigma_s,
                             a_1, a_2, a_3, a_4):

    amplitudes = np.array([a_1, a_2, a_3, a_4])
    n = np.arange(1, amplitudes.shape[0] + 1, 1)
    sigma = sigma_e ** 2 + n * sigma_s ** 2

    value = x - (n * gain + baseline)[..., np.newaxis]
    value = value ** 2
    value /= 2 * sigma[..., np.newaxis]
    pdf = np.exp(-value) * (amplitudes / np.sqrt(sigma))[..., np.newaxis]
    pdf = np.sum(pdf, axis=0)
    pdf /= np.sqrt(2 * np.pi)

    return pdf
