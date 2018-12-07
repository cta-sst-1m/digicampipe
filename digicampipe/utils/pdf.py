import numpy as np


def gaussian(x, mean, sigma, amplitude):
    x = np.atleast_1d(x)
    pdf = (x[:, np.newaxis] - mean) ** 2 / (2 * sigma ** 2)
    pdf = np.exp(-pdf)
    pdf /= (np.sqrt(2 * np.pi) * sigma)
    pdf *= amplitude

    return pdf


def generalized_poisson(k, mu, mu_xt, amplitude=1):
    """
    Reference can be found here: https://arxiv.org/pdf/math/0606238.pdf
    S. Vinogradov https://arxiv.org/pdf/1109.2014.pdf
    :param k:
    :param mu:
    :param mu_xt:
    :param amplitude:
    :return:
    """
    if isinstance(mu, np.ndarray):
        mu = mu[:, None]

    mask_valid = (mu_xt >= 0) * (mu >= 0) * (k >= 0)

    log_amplitude = np.log(amplitude)
    log_mu = np.log(mu)

    temp = np.ones((len(k), k.max()))

    temp[:] = np.arange(1, k.max() + 1)
    mask = np.triu_indices(n=temp.shape[0], m=temp.shape[1])
    temp[mask] = 1

    temp = np.log(temp)
    log_k = np.sum(temp, axis=-1)

    pdf = log_amplitude + log_mu
    pdf = pdf + np.log(mu + k * mu_xt) * (k - 1)
    pdf = pdf + (-mu - k * mu_xt) - log_k
    pdf = np.exp(pdf)
    pdf[~mask_valid] = 0

    return pdf


def mpe_distribution_general(x, bin_width, baseline, gain, sigma_e, sigma_s,
                             mu, mu_xt, amplitude, n_peaks=30):
    if n_peaks > 0:

        x = x - baseline
        photoelectron_peak = np.arange(n_peaks, dtype=np.int)
        sigma_n = sigma_e ** 2 + photoelectron_peak * sigma_s ** 2
        sigma_n = sigma_n + bin_width ** 2 / 12
        sigma_n = np.sqrt(sigma_n)

        pdf = generalized_poisson(photoelectron_peak, mu, mu_xt)

        pdf = pdf * gaussian(x, photoelectron_peak * gain, sigma_n,
                             amplitude=1)
        pdf = np.sum(pdf, axis=-1)

        return pdf * amplitude

    else:

        return 0


def fmpe_pdf_10(x, baseline, gain, sigma_e, sigma_s, bin_width, a_0=0, a_1=0,
                a_2=0, a_3=0, a_4=0, a_5=0, a_6=0, a_7=0, a_8=0, a_9=0):
    # sigma_e = np.sqrt(sigma_e**2 - 2**2 / 12)

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

    return fmpe_pdf(x, bin_width, **params)


def fmpe_pdf(x, bin_width, **params):
    baseline = params['baseline']
    sigma_e = params['sigma_e']
    sigma_s = params['sigma_s']
    gain = params['gain']

    ids = []

    for key, val in params.items():

        if key[:2] == 'a_':
            id = int(key[2:])
            ids.append(id)

    n_peaks = len(ids)

    amplitudes = np.zeros(n_peaks)

    for key, val in params.items():

        if key[:2] == 'a_':
            id = int(key[2:])
            amplitudes[id] = val

    N = np.arange(0, n_peaks, 1)
    sigma = sigma_e ** 2 + N * sigma_s ** 2 + bin_width ** 2 / 12

    value = x - (N * gain + baseline)[..., np.newaxis]
    value = value ** 2
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


def log_spe(x, baseline, gain, sigma_e, sigma_s, a_1, a_2, a_3, a_4):
    return np.log(single_photoelectron_pdf(x,
                                           baseline,
                                           gain,
                                           sigma_e,
                                           sigma_s,
                                           a_1,
                                           a_2,
                                           a_3,
                                           a_4
                                           ))


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    n = np.arange(10)
    x = np.linspace(-1000, 1000, num=10000)

    x_width = (x[-1] - x[0]) / len(x)
    gain = 10
    mu = 1
    mu_xt = 0.5
    sigma_e = 1
    sigma_s = 1
    baseline = 100
    bin_width = 10
    amplitude = 50

    pdf = mpe_distribution_general(x, bin_width, baseline, gain, sigma_e,
                                   sigma_s, mu, mu_xt, amplitude, n_peaks=50)

    plt.figure()
    plt.plot(x, pdf, label='area : {}'.format(np.sum(pdf) * x_width))
    plt.legend()
    plt.show()
