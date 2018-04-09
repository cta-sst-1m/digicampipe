import numpy as np


def gaussian(x, mean, sigma, amplitude):

    pdf = (x - mean)**2 / (2 * sigma**2)
    pdf = np.exp(-pdf)
    pdf /= np.sqrt(2 * np.pi) * sigma
    pdf *= amplitude

    return pdf


def fmpe_pdf_10(x, baseline, gain, sigma_e, sigma_s, a_0, a_1, a_2, a_3, a_4, a_5, a_6, a_7, a_8, a_9):

    params = {'baseline': baseline,
              'gain': gain,
              'sigma_e': sigma_e,
              'sigma_s': sigma_s,
              'a_0': a_0,
              'a_1': a_1,
              'a_2': a_2,
              'a_3': a_3,
              'a_4': a_4,
              'a_5': a_5,
              'a_6': a_6,
              'a_7': a_7,
              'a_8': a_8,
              'a_9': a_9}

    return fmpe_pdf(x, **params)


def fmpe_pdf(x, **params):

    baseline = params['baseline']
    sigma_e = params['sigma_e']
    sigma_s = params['sigma_s']
    gain = params['gain']

    amplitudes = []
    ids = []

    for key, val in params.items():

        if key[:2] == 'a_':

            id = int(key[2:])
            ids.append(id)
            amplitudes.append(val)

    amplitudes = np.array(amplitudes)[ids]

    N = np.arange(0, amplitudes.shape[0], 1)
    sigma = sigma_e**2 + N * sigma_s**2

    value = x - (N * gain + baseline)[..., np.newaxis]
    value = value**2
    value /= 2 * sigma[..., np.newaxis]
    temp = np.exp(-value) * (amplitudes / np.sqrt(sigma))[..., np.newaxis]
    temp = np.sum(temp, axis=0)
    temp /= np.sqrt(2 * np.pi)

    return temp


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
