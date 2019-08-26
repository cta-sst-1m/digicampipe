import numpy as np

from digicampipe.utils.pdf import fmpe_pdf_10, mpe_fit, gaussian
from digicampipe.utils.exception import PeakNotFound
from histogram.fit import HistogramFitter


class FMPEFitter(HistogramFitter):
    def __init__(self, histogram, estimated_gain, n_peaks=10, **kwargs):

        self.estimated_gain = estimated_gain
        self.n_peaks = n_peaks
        super(FMPEFitter, self).__init__(histogram, **kwargs)

        self.parameters_plot_name = {'baseline': '$B$', 'gain': 'G',
                                     'sigma_e': '$\sigma_e$',
                                     'sigma_s': '$\sigma_s$',
                                     'a_0': None, 'a_1': None, 'a_2': None,
                                     'a_3': None,
                                     'a_4': None, 'a_5': None, 'a_6': None,
                                     'a_7': None, 'a_8': None, 'a_9': None,
                                     'bin_width': None}

    def pdf(self, x, baseline, gain, sigma_e, sigma_s, a_0, a_1, a_2,
            a_3, a_4, a_5, a_6, a_7, a_8, a_9):

        params = {'baseline': baseline, 'gain': gain, 'sigma_e': sigma_e,
                  'sigma_s': sigma_s, 'a_0': a_0, 'a_1': a_1, 'a_2': a_2,
                  'a_3': a_3, 'a_4': a_4, 'a_5': a_5, 'a_6': a_6, 'a_7': a_7,
                  'a_8': a_8, 'a_9': a_9, 'bin_width': 0}

        return fmpe_pdf_10(x, **params)

    def initialize_fit(self):

        y = self.count.astype(np.float)
        x = self.bin_centers
        min_dist = self.estimated_gain / 3
        min_dist = int(min_dist)

        n_peaks = self.n_peaks

        cleaned_y = np.convolve(y, np.ones(min_dist), mode='same')
        bin_width = np.diff(x)
        bin_width = np.mean(bin_width)

        if (x != np.sort(x)).any():
            raise ValueError('x must be sorted !')

        d_y = np.diff(cleaned_y)
        indices = np.arange(len(y))
        peak_mask = np.zeros(y.shape, dtype=bool)
        peak_mask[1:-1] = (d_y[:-1] > 0) * (d_y[1:] <= 0)
        peak_mask[-min_dist:] = 0
        peak_indices = indices[peak_mask]
        peak_indices = peak_indices[:min(len(peak_indices), n_peaks)]

        if len(peak_indices) <= 1:
            raise PeakNotFound('Not enough peak found for : \n'
                               'Min distance : {} \n '
                               'Need a least 2 peaks, found {}!!'.
                               format(min_dist, len(peak_indices)))

        x_peak = x[peak_indices]
        y_peak = y[peak_indices]
        gain = np.diff(x_peak)
        weights = y_peak[:-1] ** 2
        gain = np.average(gain, weights=weights)

        sigma = np.zeros(len(peak_indices))
        mean_peak_x = np.zeros(len(peak_indices))
        amplitudes = np.zeros(len(peak_indices))

        distance = int(gain / 2)

        if distance < bin_width:
            raise ValueError(
                'Distance between peaks must be >= {} the bin width'
                ''.format(bin_width))

        n_x = len(x)

        for i, peak_index in enumerate(peak_indices):
            left = x[peak_index] - distance
            left = np.searchsorted(x, left)
            left = max(0, left)
            right = x[peak_index] + distance + 1
            right = np.searchsorted(x, right)
            right = min(n_x - 1, right)

            amplitudes[i] = np.sum(y[left:right]) * bin_width
            mean_peak_x[i] = np.average(x[left:right], weights=y[left:right])

            sigma[i] = np.average((x[left:right] - mean_peak_x[i]) ** 2,
                                  weights=y[left:right])
            sigma[i] = np.sqrt(sigma[i] - bin_width ** 2 / 12)

        gain = np.diff(mean_peak_x)
        weights = None
        # weights = amplitudes[:-1] ** 2
        gain = np.average(gain, weights=weights)

        sigma_e = np.sqrt(sigma[0] ** 2)
        sigma_s = (sigma[1:] ** 2 - sigma_e ** 2) / np.arange(1, len(sigma), 1)
        sigma_s = np.mean(sigma_s)

        if sigma_s < 0:
            sigma_s = sigma_e ** 2

        sigma_s = np.sqrt(sigma_s)

        params = {'baseline': mean_peak_x[0], 'sigma_e': sigma_e,
                  'sigma_s': sigma_s, 'gain': gain}

        for i in range(n_peaks):

            if i < len(amplitudes):

                value = amplitudes[i]

            else:

                value = amplitudes.min()

            params['a_{}'.format(i)] = value

        self.initial_parameters = params

        return params

    def compute_fit_boundaries(self):

        limit_params = {}
        init_params = self.initial_parameters

        baseline = init_params['baseline']
        gain = init_params['gain']
        sigma_e = init_params['sigma_e']
        sigma_s = init_params['sigma_s']

        limit_params['limit_baseline'] = (baseline - sigma_e,
                                          baseline + sigma_e)
        limit_params['limit_gain'] = (0.5 * gain, 1.5 * gain)
        limit_params['limit_sigma_e'] = (0.5 * sigma_e, 1.5 * sigma_e)
        limit_params['limit_sigma_s'] = (0.5 * sigma_s, 1.5 * sigma_s)

        for key, val in init_params.items():

            if key[:2] == 'a_':
                limit_params['limit_{}'.format(key)] = (0.5 * val, val * 1.5)

        self.boundary_parameter = limit_params

        return limit_params

    def compute_data_bounds(self):

        x = self.histogram.bin_centers
        bin_width = np.diff(self.histogram.bins)
        y = self.histogram.data
        if not self.parameters:
            params = self.initial_parameters

        else:
            params = self.parameters

        n_peaks = self.n_peaks

        mask = (y > 0) * (x < n_peaks * self.estimated_gain)

        if 'gain' in params.keys() and 'baseline' in params.keys():

            gain = params['gain']
            baseline = params['baseline']
            amplitudes = []

            for key, val in params.items():

                if key[:2] == 'a_':
                    amplitudes.append(val)

            amplitudes = np.array(amplitudes)
            amplitudes = amplitudes[amplitudes > 0]
            n_peaks = len(amplitudes)

            min_bin = baseline - gain / 2
            max_bin = baseline + gain * (n_peaks - 1)
            max_bin += gain / 2

            mask *= (x <= max_bin) * (x >= min_bin)

        return x[mask], y[mask], bin_width[mask]


class MaxHistoFitter(FMPEFitter):
    def __init__(self, histogram, estimated_gain, **kwargs):
        n_peaks = 2
        super(MaxHistoFitter, self).__init__(histogram, estimated_gain,
                                             n_peaks, **kwargs)
        self.parameters_plot_name = {'baseline': '$B$', 'gain': 'G',
                                     'sigma_e': '$\sigma_e$',
                                     'sigma_s': '$\sigma_s$',
                                     'a_0': None, 'a_1': None}

    def pdf(self, x, baseline, gain, sigma_e, sigma_s, a_0, a_1):
        params = {'baseline': baseline, 'gain': gain, 'sigma_e': sigma_e,
                  'sigma_s': sigma_s, 'a_0': a_0, 'a_1': a_1, 'bin_width': 0}

        return fmpe_pdf_10(x, **params)


class SPEFitter(FMPEFitter):
    def __init__(self, histogram, estimated_gain, **kwargs):
        n_peaks = 4
        super(SPEFitter, self).__init__(histogram, estimated_gain, n_peaks,
                                        **kwargs)
        self.parameters_plot_name = {'baseline': '$B$', 'gain': 'G',
                                     'sigma_e': '$\sigma_e$',
                                     'sigma_s': '$\sigma_s$',
                                     'a_1': None, 'a_2': None, 'a_3': None,
                                     'a_4': None}

    def pdf(self, x, baseline, gain, sigma_e, sigma_s, a_1, a_2, a_3, a_4):
        params = {'baseline': baseline, 'gain': gain, 'sigma_e': sigma_e,
                  'sigma_s': sigma_s, 'a_0': 0, 'a_1': a_1, 'a_2': a_2,
                  'a_3': a_3, 'a_4': a_4, 'bin_width': 0}

        return fmpe_pdf_10(x, **params)

    def initialize_fit(self):
        init_params = super(SPEFitter, self).initialize_fit()

        init_params['a_4'] = init_params['a_3']
        init_params['a_3'] = init_params['a_2']
        init_params['a_2'] = init_params['a_1']
        init_params['a_1'] = init_params['a_0']

        init_params['baseline'] = init_params['baseline'] - init_params['gain']

        del init_params['a_0']

        self.initial_parameters = init_params

        return init_params


class MPEFitter(HistogramFitter):
    def __init__(self, histogram, fixed_params, **kwargs):

        super(MPEFitter, self).__init__(histogram, **kwargs)
        self.initial_parameters = fixed_params
        self.iminuit_options = {**self.iminuit_options, **fixed_params}
        self.parameters_plot_name = {'mu': '$\mu$', 'mu_xt': '$\mu_{XT}$',
                                     'n_peaks': '$N_{peaks}$', 'gain': '$G$',
                                     'amplitude': '$A$', 'baseline': '$B$',
                                     'sigma_e': '$\sigma_e$',
                                     'sigma_s': '$\sigma_s$'
                                     }

    def initialize_fit(self):

        fixed_params = self.initial_parameters
        x = self.bin_centers
        y = self.count

        gain = fixed_params['gain']
        sigma_e = fixed_params['sigma_e']
        sigma_s = fixed_params['sigma_s']
        baseline = fixed_params['baseline']

        mean_x = np.average(x, weights=y) - baseline

        if 'mu_xt' in fixed_params.keys():

            mu_xt = fixed_params['mu_xt']
            mu = mean_x * (1 - mu_xt) / gain

        else:

            left = baseline - gain / 2
            left = np.where(x > left)[0][0]

            right = baseline + gain / 2
            right = np.where(x < right)[0][-1]

            probability_0_pe = np.sum(y[left:right])
            probability_0_pe /= np.sum(y)
            mu = - np.log(probability_0_pe)

            mu_xt = 1 - gain * mu / mean_x
            mu_xt = max(0.01, mu_xt)

        n_peaks = np.max(x) - (baseline - gain / 2)
        n_peaks = n_peaks / gain
        n_peaks = np.round(n_peaks)
        amplitude = np.sum(y)

        params = {'baseline': baseline, 'sigma_e': sigma_e,
                  'sigma_s': sigma_s, 'gain': gain, 'amplitude': amplitude,
                  'mu': mu, 'mu_xt': mu_xt, 'n_peaks': n_peaks}

        self.initial_parameters = params

    def compute_fit_boundaries(self):

        limit_params = {}

        init_params = self.initial_parameters

        baseline = init_params['baseline']
        gain = init_params['gain']
        sigma_e = init_params['sigma_e']
        sigma_s = init_params['sigma_s']
        mu = init_params['mu']
        amplitude = init_params['amplitude']
        n_peaks = init_params['n_peaks']

        limit_params['limit_baseline'] = (
            baseline - sigma_e, baseline + sigma_e)
        limit_params['limit_gain'] = (0.5 * gain, 1.5 * gain)
        limit_params['limit_sigma_e'] = (0.5 * sigma_e, 1.5 * sigma_e)
        limit_params['limit_sigma_s'] = (0.5 * sigma_s, 1.5 * sigma_s)
        limit_params['limit_mu'] = (0.5 * mu, 1.5 * mu)
        limit_params['limit_mu_xt'] = (0, 0.5)
        limit_params['limit_amplitude'] = (0.5 * amplitude, 1.5 * amplitude)
        limit_params['limit_n_peaks'] = (max(1., n_peaks - 1.), n_peaks + 1.)

        self.boundary_parameter = limit_params

    def pdf(self, x, baseline, gain, sigma_e, sigma_s, mu, mu_xt, amplitude,
            n_peaks):

        return mpe_fit(x, baseline, gain, sigma_e, sigma_s, mu, mu_xt, amplitude,
            n_peaks)


class MPECombinedFitter(HistogramFitter):

    def __init__(self, histogram, n_peaks, **kwargs):

        super(MPECombinedFitter, self).__init__(histogram, **kwargs)

        self.n_peaks = n_peaks
        self.histo_mean = histogram.mean()
        self.amplitude = histogram.data.sum(axis=-1)
        if isinstance(self.amplitude, np.ndarray):

            self.amplitude = self.amplitude[:, np.newaxis]
        # self.mask = mask

        self.parameters_plot_name = {'mu_xt': '$\mu_{XT}$',
                                     'gain': '$G$', 'baseline': '$B$',
                                     'sigma_e': '$\sigma_e$',
                                     'sigma_s': '$\sigma_s$',
                                     'mu': '$\mu$',
                                     }

    def compute_data_bounds(self):

        return self.histogram.bin_centers, self.histogram.data, np.diff(self.histogram.bins)

    def initialize_fit(self):

        pass

    def compute_fit_boundaries(self):

        pass

    def pdf(self, x, baseline, gain, sigma_e, sigma_s, mu_xt, **kwargs):

        if 'mu' not in kwargs.keys():

            mu = (self.histo_mean - baseline) / gain
            mu *= (1 - mu_xt)

        else:

            mu = kwargs['mu']

        # scale = (1 - mu_xt)
        scale = 1
        shift = - gain * (1 - 1 / (1 - mu_xt))
        shift = 0
        y = mpe_fit(x / scale + shift, baseline=baseline, gain=gain,
                    sigma_e=sigma_e,
                    sigma_s=sigma_s, mu=mu, mu_xt=mu_xt, amplitude=1,
                    n_peaks=self.n_peaks) * self.amplitude / scale

        return y


class GaussianFitter(HistogramFitter):

    def initialize_fit(self):

        x = self.bin_centers
        y = self.count

        mean = np.average(x, weights=y)
        std = np.average((x - mean) ** 2, weights=y)
        std = np.sqrt(std)
        amplitude = np.sum(y, dtype=np.float)

        self.initial_parameters = {'mean': mean, 'sigma': std,
                                   'amplitude': amplitude}

    def compute_data_bounds(self):

        mask = self.histogram.data > 0

        return self.histogram.bin_centers[mask], self.histogram.data[mask], np.diff(self.histogram.bins)[mask]

    def compute_fit_boundaries(self):

        bounds = {}

        for key, val in self.initial_parameters.items():

            if val > 0:
                bounds['limit_' + key] = (val * 0.5, val * 1.5)

            else:

                bounds['limit_' + key] = (val * 1.5, val * 0.5)

        self.boundary_parameter = bounds

    def pdf(self, x, mean, sigma, amplitude):

        pdf = (x - mean) / (np.sqrt(2) * sigma)
        pdf = - pdf ** 2
        pdf = np.exp(pdf)
        pdf = pdf * amplitude / (sigma * np.sqrt(2 * np.pi))

        return pdf

    def log_pdf(self, x, mean, sigma, amplitude):

        temp = np.log(amplitude) - np.log(sigma * np.sqrt(2 * np.pi))
        temp = temp - ((x - mean) / (sigma * np.sqrt(2))) ** 2

        return temp
