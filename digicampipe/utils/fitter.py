from abc import abstractmethod, ABC
import numpy as np
from scipy.optimize import minimize
from pkg_resources import resource_filename
import os
import inspect
from digicampipe.utils.pulse_template import NormalizedPulseTemplate
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from copy import copy
from digicampipe.instrument.camera import DigiCam
from digicampipe.utils.pdf import gaussian2d
from scipy.ndimage import convolve1d
from ctapipe.image import cleaning, hillas_parameters
from ctapipe.image.cleaning import apply_time_delta_cleaning, dilate
from ctapipe.image.timing_parameters import timing_parameters
from digicampipe.visualization.plot import plot_array_camera
from digicampipe.utils.pdf import mpe_distribution_general, log_gaussian, log_generalized_poisson, log_generalized_poisson_1d, log_gaussian2d
from digicampipe.image.hillas import compute_alpha
from digicampipe.image.disp import compute_leakage
from matplotlib.patches import Arrow
from scipy.special import gammaln
from numpy.ctypeslib import ndpointer
import ctypes
from digicampipe.io.containers import ImageParametersContainer

from iminuit import Minuit

GEOMETRY = DigiCam.geometry

TEMPLATE_FILENAME = resource_filename(
    'digicampipe',
    os.path.join(
        'tests',
        'resources',
        'pulse_SST-1M_pixel_0.dat'
    )
)


"""
lib = np.ctypeslib.load_library("pixel_likelihood", os.path.dirname(__file__))

image_loglikelihood = lib.image_loglikelihood
image_loglikelihood.argtypes = [
    ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
    ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
    ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
    ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
    ctypes.c_uint,
    ctypes.c_uint,
    ctypes.c_double,
    ctypes.c_double,
    ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
    ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
    ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
    ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
    ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
    ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
    ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
    ctypes.c_uint,
    ctypes.c_uint,
    ctypes.c_double,
    ctypes.c_double,
    ctypes.c_double,
    ctypes.c_double,
    ctypes.c_double,
    ctypes.c_double,
    ctypes.c_double,
    ctypes.c_double,
]
image_loglikelihood.restype = ctypes.c_double
"""

class Fitter(ABC):

    def __init__(self, data):

        self.data = data
        self.end_parameters = None
        self.start_parameters = None
        self.bounds = None
        self.names_parameters = list(inspect.signature(self.log_pdf).parameters)
        self.start_parameters = self.initialize_fit()
        self.bounds = self.compute_bounds()
        self.error_parameters = None
        self.correlation_matrix = None

    def __str__(self):

        str = 'Start parameters :\n\t{}\n'.format(self.start_parameters)
        str += 'Bound parameters :\n\t{}\n'.format(self.bounds)
        str += 'End parameters :\n\t{}\n'.format(self.end_parameters)
        str += 'Error parameters :\n\t{}\n'.format(self.error_parameters)
        str += 'Log-Likelihood :\t{}'.format(self.log_likelihood(**self.end_parameters))

        return str

    def fit(self, verbose=True, minuit=False, **kwargs):

        if minuit:

            fixed_params = {}
            bounds_params = {}
            start_params = self.start_parameters

            for key, val in self.bounds.items():

                bounds_params['limit_'+ key] = val

            for key in self.names_parameters:

                if key in kwargs.keys():

                    fixed_params['fix_' + key] = True

                else:

                    fixed_params['fix_' + key] = False

            # print(fixed_params, bounds_params, start_params)
            options = {**start_params, **bounds_params, **fixed_params}
            f = lambda *args: -self.log_likelihood(*args)
            print_level = 2 if verbose else 0
            m = Minuit(f, print_level=print_level, forced_parameters=self.names_parameters, errordef=0.5, **options)
            m.migrad()
            self.end_parameters = dict(m.values)
            options = {**self.end_parameters, **fixed_params}
            m = Minuit(f, print_level=print_level, forced_parameters=self.names_parameters, errordef=0.5, **options)
            m.migrad()
            try:
                self.error_parameters = dict(m.errors)

            except (KeyError, AttributeError, RuntimeError):

                self.error_parameters = {key: np.nan for key in self.names_parameters}
                pass
            # print(self.end_parameters, self.error_parameters)

        else:

            fixed_params = {}

            for param in self.names_parameters:
                if param in kwargs.keys():
                    fixed_params[param] = kwargs[param]
                    del kwargs[param]

            start_parameters = []
            bounds = []
            name_parameters = []

            for key in self.names_parameters:

                if key not in fixed_params.keys():

                    start_parameters.append(self.start_parameters[key])
                    bounds.append(self.bounds[key])
                    name_parameters.append(key)

            def llh(x):

                params = dict(zip(name_parameters, x))
                return -self.log_likelihood(**params, **fixed_params)

            result = minimize(llh, x0=start_parameters, bounds=bounds, **kwargs)
            self.end_parameters = dict(zip(name_parameters, result.x))
            self.end_parameters.update(fixed_params)

            try:
                self.correlation_matrix = result.hess_inv.todense()
                self.error_parameters = dict(zip(name_parameters,
                                             np.diagonal(np.sqrt(self.correlation_matrix))))
            except (KeyError, AttributeError):
                pass

            if verbose:

                print(result)

    def pdf(self, *args, **kwargs):

        return np.exp(self.log_pdf(*args, **kwargs))

    @abstractmethod
    def plot(self):

        pass

    @abstractmethod
    def compute_bounds(self):

        pass

    @abstractmethod
    def initialize_fit(self):

        pass

    @abstractmethod
    def log_pdf(self, *args, **kwargs):

        pass

    def likelihood(self, *args, **kwargs):

        return np.exp(self.log_likelihood(*args, **kwargs))

    def log_likelihood(self, *args, **kwargs):

        llh = self.log_pdf(*args, **kwargs)
        return np.sum(llh)

    def plot_1dlikelihood(self, parameter_name, axes=None, size=1000,
                        x_label=None, invert=False):

        key = parameter_name

        if key not in self.names_parameters:

            raise NameError('Parameter : {} not in existing parameters :'
                            '{}'.format(key, self.names_parameters))

        x = np.linspace(self.bounds[key][0], self.bounds[key][1], num=size)
        if axes is None:

            fig = plt.figure()
            axes = fig.add_subplot(111)

        params = copy(self.end_parameters)
        llh = np.zeros(x.shape)

        for i, xx in enumerate(x):

            params[key] = xx
            llh[i] = self.log_likelihood(**params)

        x_label = self.labels[key] if x_label is None else x_label

        if not invert:
            axes.plot(x, -llh, color='r')
            axes.axvline(self.end_parameters[key], linestyle='--', color='k',
                         label='Fitted value {:.2f}'.format(
                             self.end_parameters[key]))
            axes.axvline(self.start_parameters[key], linestyle='--',
                         color='b', label='Starting value {:.2f}'.format(
                    self.start_parameters[key]
                ))
            #axes.axvspan(self.bounds[key][0],
            #             self.bounds[key][1], label='bounds',
            #             alpha=0.5, facecolor='k')
            axes.set_ylabel('-$\ln \mathcal{L}$')
            axes.set_xlabel(x_label)

        else:

            axes.plot(-llh, x, color='r')
            axes.axhline(self.end_parameters[key], linestyle='--',
                         color='k',
                         label='Fitted value {:.2f}'.format(
                             self.end_parameters[key]))
            axes.axhline(self.start_parameters[key], linestyle='--',
                         color='b', label='Starting value {:.2f}'.format(
                    self.start_parameters[key]
                ))
            axes.axhspan(self.bounds[key][0],
                         self.bounds[key][1], label='bounds',
                         alpha=0.5, facecolor='k')
            axes.set_xlabel('-$\ln \mathcal{L}$')
            axes.set_ylabel(x_label)
            axes.xaxis.set_label_position('top')
            # axes.xaxis.tick_top()

        axes.legend(loc='best')
        return axes

    def plot_2dlikelihood(self, parameter_1, parameter_2=None, size=100,
                          x_label=None, y_label=None):

        if isinstance(size, int):
            size = (size, size)

        key_x = parameter_1
        key_y = parameter_2
        x = np.linspace(self.bounds[key_x], self.bounds[key_x], num=size[0])
        y = np.linspace(self.bounds[key_y], self.bounds[key_y], num=size[1])
        dx = x[1] - x[0]
        dy = y[1] - y[0]

        params = copy(self.end_parameters)
        llh = np.zeros(size)

        for i, xx in enumerate(x):
            params[key_x] = xx
            for j, yy in enumerate(y):

                params[key_y] = yy
                llh[i, j] = self.log_likelihood(**params)


        fig = plt.figure()
        left, width = 0.1, 0.65
        bottom, height = 0.1, 0.65
        spacing = 0.005
        rect_center = [left, bottom, width, height]
        rect_x = [left, bottom + height + spacing, width, 0.2]
        rect_y = [left + width + spacing, bottom, 0.2, height]
        axes = fig.add_axes(rect_center)
        axes_x = fig.add_axes(rect_x)
        axes_y = fig.add_axes(rect_y)
        axes.tick_params(direction='in', top=True, right=True)
        self.plot_1dlikelihood(parameter_name=parameter_1, axes=axes_x)
        self.plot_1dlikelihood(parameter_name=parameter_2, axes=axes_y,
                               invert=True)
        axes_x.tick_params(direction='in', labelbottom=False)
        axes_y.tick_params(direction='in', labelleft=False)

        axes_x.set_xlabel('')
        axes_y.set_ylabel('')
        axes_x.tick_params(
                axis='x',          # changes apply to the x-axis
                which='both',      # both major and minor ticks are affected
                bottom=False,      # ticks along the bottom edge are off
                top=False,         # ticks along the top edge are off
                labelbottom=False)
        axes_y.tick_params(
                axis='y',          # changes apply to the x-axis
                which='both',      # both major and minor ticks are affected
                bottom=False,      # ticks along the bottom edge are off
                top=False,         # ticks along the top edge are off
                labelbottom=False)

        x_label = self.labels[key_x] if x_label is None else x_label
        y_label = self.labels[key_y] if y_label is None else y_label

        im = axes.imshow(-llh.T,  origin='lower', extent=[x.min() - dx/2.,
                                                     x.max() - dx/2,
                                                     y.min() - dy/2,
                                                     y.max() - dy/2], aspect='auto')

        axes.scatter(self.end_parameters[key_x], self.end_parameters[key_y],
                     marker='x', color='w', label='Maximum Likelihood')
        axes.set_xlabel(x_label)
        axes.set_ylabel(y_label)
        axes.legend(loc='best')
        plt.colorbar(mappable=im, ax=axes_y, label='-$\ln \mathcal{L}$')
        axes_x.set_xlim(x.min(), x.max())
        axes_y.set_ylim(y.min(), y.max())

        return axes

    def plot_likelihood(self, parameter_1, parameter_2=None,
                        axes=None, size=1000,
                          x_label=None, y_label=None):

        if parameter_2 is None:

            return self.plot_1dlikelihood(parameter_name=parameter_1,
                                          axes=axes, x_label=x_label, size=size)

        else:

            return self.plot_2dlikelihood(parameter_1,
                                          parameter_2=parameter_2,
                                          size=size,
                                          x_label=x_label,
                                          y_label=y_label)


class PulseTemplateFitter(Fitter):

    def __init__(self, data,
                 template=NormalizedPulseTemplate.load(TEMPLATE_FILENAME),
                 error=None, dt=4):
        if len(data.shape) > 1:

            raise ValueError('Invalid shape for data')

        n_samples = len(data)
        self.times = np.arange(0, n_samples) * dt
        self.template = template
        self.error = error if error is not None else np.ones(n_samples)
        self.labels = {'charge': 'Charge [LSB]',
                       't_0': '$t_{0}$ [ns]',
                       }
        super().__init__(data)

    def log_pdf(self, charge, baseline, t_0, sigma_t):

        # norm = np.sqrt(1 / (2 * np.pi)) / self.error
        norm = - 0.5 * np.log(2*np.pi) - np.log(self.error)
        y_fit = charge * self.template((self.times - t_0) / sigma_t) + baseline
        log_pdf = - 0.5 * ((y_fit - self.data) / self.error) ** 2 + norm

        return log_pdf

    def initialize_fit(self):

        baseline = np.min(self.data, axis=-1)
        index_max = np.argmax(self.data, axis=-1)
        shift = self.template.compute_time_of_max()
        t_0 = self.times[index_max] - shift
        charge = self.data[:][index_max] - baseline
        sigma_t = 1

        params = dict(zip(self.names_parameters, [charge, baseline, t_0, sigma_t]))

        return params

    def compute_bounds(self):

        charge = (0, self.start_parameters['charge'] * 2)
        baseline = (self.data.min(axis=-1) * 0.5, self.data.max(axis=-1) * 1.5)
        t_0 = (self.times.min() - 50, self.times.max() + 50)
        sigma_t = (1, 10)
        params = dict(zip(self.names_parameters, [charge, baseline, t_0, sigma_t]))

        return params

    def plot(self, axes=None, **kwargs):

        if axes is None:

            fig = plt.figure()
            axes = fig.add_subplot(111)

        t = np.linspace(self.times.min(), self.times.max(),
                        num=len(self.times)*100)
        y = self.template((t - self.end_parameters['t_0'])/self.end_parameters['sigma_t'],
                                   amplitude=self.end_parameters['charge'],
                                   baseline=self.end_parameters['baseline'])
        axes.errorbar(self.times, self.data, yerr=self.error,
                  color='k', marker='o', label='Data',
                  linestyle='None')
        axes.plot(t, y, color='r', label='Fit')
        axes.set_xlabel('time [ns]')
        axes.set_ylabel('[LSB]')
        axes.legend(loc='best')

        return axes


class MPEPulseTemplateFitter(Fitter):

    def __init__(self, data,
                 gain,
                 mu_xt,
                 sigma_e,
                 sigma_s,
                 baseline,
                 template=NormalizedPulseTemplate.load(TEMPLATE_FILENAME),
                 error=None, dt=4):
        if len(data.shape) > 1:

            raise ValueError('Invalid shape for data')

        self.n_samples = len(data)
        self.dt = dt
        self.times = np.arange(0, self.n_samples) * self.dt
        self.template = template
        self.error = error if error is not None else np.ones(self.n_samples)

        self.gain = gain
        self.mu_xt = mu_xt
        self.sigma_e = sigma_e
        self.sigma_s = sigma_s
        self.baseline = baseline

        self.labels = {'n_pe': '$N_{p.e.}$ [p.e.]',
                       't_0': '$t_{0}$ [ns]',
                       }

        super().__init__(data)

    def log_pdf(self, n_pe, t_0, sigma_t):

        charge = n_pe * self.gain / (1 - self.mu_xt)
        log_pdf = - 0.5 * self.n_samples * np.log(2*np.pi)
        y_fit = charge * self.template((self.times - t_0) / sigma_t) + self.baseline
        log_pdf = log_pdf - 0.5 * np.sum(((y_fit - self.data) / self.error) ** 2)
        n_peaks = int(n_pe * 10)
        prob_pe = mpe_distribution_general(charge + self.baseline, 0, self.baseline,
                                           self.gain, self.sigma_e,
                                           self.sigma_s, n_pe,
                                           self.mu_xt, 1, n_peaks)
        prob_pe = np.log(prob_pe)

        if not np.isfinite(prob_pe):
            prob_pe = - 1E8

        log_pdf = log_pdf + prob_pe

        return log_pdf

    def initialize_fit(self):

        index_max = np.argmax(self.data, axis=-1)
        shift = self.template.compute_time_of_max()
        t_0 = self.times[index_max] - shift
        charge = self.data[..., index_max]
        n_pe = (charge - self.baseline) / self.gain * (1 - self.mu_xt)
        sigma_t = 1

        params = dict(zip(self.names_parameters, [n_pe, t_0, sigma_t]))

        return params

    def compute_bounds(self):

        n_pe = (0, self.start_parameters['n_pe'] * 2)
        t_0 = (self.times.min() - 50, self.times.max() + 50)
        sigma_t = (1, 10)
        params = dict(zip(self.names_parameters, [n_pe, t_0, sigma_t]))

        return params

    def plot(self, axes=None, **kwargs):

        if axes is None:

            fig = plt.figure()
            axes = fig.add_subplot(111)

        t = np.linspace(self.times.min(), self.times.max(),
                        num=len(self.times)*100)
        charge = self.end_parameters['n_pe'] * self.gain / (1 - self.mu_xt)
        y = self.template((t - self.end_parameters['t_0'])/self.end_parameters['sigma_t'],
                                   amplitude=charge,
                                   baseline=self.baseline)
        axes.errorbar(self.times, self.data, yerr=self.error,
                  color='k', marker='o', label='Data',
                  linestyle='None')
        axes.plot(t, y, color='r', label='Fit')
        axes.set_xlabel('time [ns]')
        axes.set_ylabel('[LSB]')
        axes.legend(loc='best')

        return axes


class MPETimeFitter(MPEPulseTemplateFitter):

    def __init__(self, data,
                 gain,
                 mu_xt,
                 sigma_e,
                 sigma_s,
                 baseline,
                 template=NormalizedPulseTemplate.load(TEMPLATE_FILENAME),
                 error=None, dt=4, n_peaks=1000):

        super().__init__(data=data,
                         gain=gain,
                 mu_xt=mu_xt,
                 sigma_e=sigma_e,
                 sigma_s=sigma_s,
                 baseline=baseline,
                 template=template,
                 error=error, dt=dt)
        self._initialize_pdf(n_peaks=n_peaks)

    def _initialize_pdf(self, n_peaks):


        photoelectron_peak = np.arange(n_peaks, dtype=np.int)
        sigma_n = self.sigma_e ** 2 + photoelectron_peak * self.sigma_s ** 2
        sigma_n = np.sqrt(sigma_n)
        self.sigma_n = sigma_n

        self.photo_peaks = photoelectron_peak

    def log_pdf(self, n_pe, t_0, sigma_t):

        """
        template = self.template((self.times -  t_0) /sigma_t)
        mask = (template > 0)
        template = template[mask]
        n_pe = n_pe * template

        log_poisson = log_generalized_poisson_1d(self.photo_peaks, n_pe, self.mu_xt)
        x = self.gain * self.photo_peaks * template[..., None] + self.baseline
        x = -(self.data[mask][..., None] - x)
        log_gauss = log_gaussian(0, x, self.sigma_n)
        log_gauss = np.squeeze(log_gauss)

        pdf = np.exp(log_poisson + log_gauss)
        pdf = np.sum(pdf, axis=-1)

        return np.log(pdf)
        """

        template = self.template((self.times - t_0) / sigma_t)
        # mask = (template > 0)
        # template = template[mask]
        # data = self.data[mask][..., None]
        data = self.data[..., None]

        log_poisson = log_generalized_poisson_1d(self.photo_peaks, n_pe, self.mu_xt)
        # sigma = np.sqrt(self.sigma_n**2 - self.sigma_e**2)

        x = self.gain * self.photo_peaks * template[..., None] + self.baseline
        x = -(data - x)

        log_gauss = log_gaussian(0, x, self.sigma_n)
        log_gauss = np.squeeze(log_gauss)

        pdf = np.exp(log_poisson + log_gauss)
        pdf = np.sum(pdf, axis=-1)

        return np.log(pdf)


class ImageFitter(Fitter):

    def __init__(self, data, error, gain, baseline, crosstalk, sigma_s, geometry,
                 dt, integral_width, template, sigma_space=4, sigma_time=3,
                 sigma_amplitude=3, picture_threshold=15, boundary_threshold=10,
                 time_before_shower=10, time_after_shower=50):

        self.dt = dt
        self.integral_width = integral_width
        self.n_pixels, self.n_samples = data.shape
        self.times = np.arange(0, self.n_samples) * self.dt
        self.error = error
        self.geometry = geometry
        self.pix_x = geometry.pix_x.value
        self.pix_y = geometry.pix_y.value
        self.pix_area = geometry.pix_area
        self.gain = gain
        self.baseline = baseline
        self.crosstalk = crosstalk
        self.sigma_s = sigma_s
        self.labels = {'charge': 'Charge [p.e.]',
                       't_cm': '$t_{CM}$ [ns]',
                       'x_cm': '$x_{CM}$ [mm]',
                       'y_cm': '$y_{CM}$ [mm]',
                       'width': '$\sigma_w$ [mm]',
                       'length': '$\sigma_l$ [mm]',
                       'psi': '$\psi$ [rad]',
                       'v': '$v$ [mm/ns]'
                       }

        self.template = template
        self.template_time_max = template.compute_time_of_max()
        self.template_gain_factor = template.compute_charge_amplitude_ratio(
            self.integral_width, self.dt)

        self.sigma_amplitude = sigma_amplitude
        self.sigma_space = sigma_space
        self.sigma_time = sigma_time
        self.picture_threshold = picture_threshold
        self.boundary_threshold = boundary_threshold
        self.time_before_shower = time_before_shower
        self.time_after_shower = time_after_shower

        super().__init__(data)

        self.mask_pixel, self.mask_time = self._clean_data()
        pixels = np.arange(self.n_pixels)[~self.mask_pixel]
        t = np.arange(self.n_samples)[~self.mask_time]
        self.data = np.delete(self.data, pixels, axis=0)
        self.data = np.delete(self.data, t, axis=1)
        self.error = np.delete(self.error, pixels, axis=0)
        self.error = np.delete(self.error, t, axis=1)
        self.times = self.times[self.mask_time]
        self.gain = self.gain[self.mask_pixel]
        self.crosstalk = self.crosstalk[self.mask_pixel]
        self.sigma_s = self.sigma_s[self.mask_pixel]
        self.baseline = self.baseline[self.mask_pixel]
        self.pix_x = self.pix_x[self.mask_pixel]
        self.pix_y = self.pix_y[self.mask_pixel]
        self.pix_area = self.pix_area[self.mask_pixel]

    def plot(self, n_sigma=3):

        charge = np.ma.masked_array(self._init_charge, mask=~self._init_mask)
        # charge = self._init_charge

        cam_display, _ = plot_array_camera(data=self._init_charge, label='$N_{p.e.}$')

        plot_array_camera(data=charge, label='$N_{p.e.}$')

        length = n_sigma * self.end_parameters['length']
        psi = self.end_parameters['psi']
        dx = length * np.cos(psi)
        dy = length * np.sin(psi)
        direction_arrow = Arrow(x=self.end_parameters['x_cm'],
                                y=self.end_parameters['y_cm'],
                                dx=dx, dy=dy, width=10, color='k',
                                label='EAS direction')

        cam_display.axes.add_patch(direction_arrow)

        cam_display.add_ellipse(centroid=(self.end_parameters['x_cm'],
                                          self.end_parameters['y_cm']),
                                width=n_sigma * self.end_parameters['width'],
                                length=length,
                                angle=psi,
                                linewidth=6, color='r', linestyle='--',
                                label='{} $\sigma$ contour'.format(n_sigma))
        cam_display.axes.legend(loc='best')

        return cam_display

    def _compute_alpha(self):

        psi = self.end_parameters['psi']
        phi = np.arctan(self.end_parameters['y_cm'] / self.end_parameters['x_cm'])
        alpha = compute_alpha(phi, psi)

        return alpha

    def _compute_charge_and_time(self):


        integral_width = self.integral_width

        charge = convolve1d(
            self.data,
            np.ones(integral_width),
            axis=-1
        )
        charge = charge * self.template.compute_charge_amplitude_ratio(
            integral_width=integral_width, dt_sampling=4)
        index_max = np.argmax(charge, axis=-1)
        charge = np.max(charge, axis=-1)
        times = self.times[index_max] - self.template_time_max

        charge = (charge - self.baseline) / self.gain * (1 - self.crosstalk)
        # charge[charge < 0] = 0

        return charge, times

    def _clean_image(self, charge, times):

        mask = cleaning.tailcuts_clean(geom=self.geometry,
                                       image=charge,
                                       picture_thresh=self.picture_threshold,
                                       boundary_thresh=self.boundary_threshold,
                                       keep_isolated_pixels=False,
                                       min_number_picture_neighbors=2)

        n_islands, islands = cleaning.number_of_islands(self.geometry, mask)
        n_pixels_per_islands = np.bincount(islands.astype(int))
        largest_island = np.argmax(n_pixels_per_islands[1:]) + 1
        mask = (islands == largest_island)
        mask = cleaning.dilate(self.geometry, mask)

        return mask

    def initialize_fit(self):

        charge, times = self._compute_charge_and_time()

        # plot_array_camera(charge)
        mask = self._clean_image(charge, times)

        self._init_mask = mask
        self._init_charge = charge
        charge[~mask] = 0
        times = np.ma.masked_array(times, mask=mask)
        hillas = hillas_parameters(self.geometry, charge)
        timing = timing_parameters(self.geometry, charge, times, hillas)

        size = hillas.intensity
        t_cm = timing.intercept
        x_cm = hillas.x.value
        y_cm = hillas.y.value
        width = max(hillas.width.value, np.sqrt(self.pix_area[0] / np.pi))
        length = hillas.length.value
        psi = hillas.psi.value
        v = timing.slope.value

        if psi < 0:

            psi = psi + np.pi
            v = -v

        params = [size, t_cm, x_cm, y_cm, width, length, psi, v]
        params = dict(zip(self.names_parameters, params))

        return params

    def _clean_data(self):


        x_cm = self.start_parameters['x_cm']
        y_cm = self.start_parameters['y_cm']
        width = self.start_parameters['width']
        length = self.start_parameters['length']
        psi = self.start_parameters['psi']

        dx = self.pix_x - x_cm
        dy = self.pix_y - y_cm

        lon = dx * np.cos(psi) + dy * np.sin(psi)
        lat = dx * np.sin(psi) - dy * np.cos(psi)

        mask_pixel = ((lon / length)**2 + (lat / width)**2) < self.sigma_space**2
        # mask_pixel = dilate(self.geometry, mask_pixel)

        v = np.abs(self.start_parameters['v'])
        t_start = self.start_parameters['t_cm'] - (v * length / 2 * self.sigma_time) - self.time_before_shower
        t_end = self.start_parameters['t_cm'] + (v * length / 2 * self.sigma_time) + self.time_after_shower

        # print(t_start, t_end, 'HELLO')
        mask_time = (self.times < t_end) * (self.times > t_start)

        # print(t_start, self.start_parameters['t_cm'], length, t_end)
        return mask_pixel, mask_time

    def compute_bounds(self, method='limit'):

        params = self.start_parameters
        bounds = {}

        if method == 'limit':
            maximum_length = 5000
            readout_max = 200
            v_min = 1 / 200  # 1 mm / per ns
            bounds['charge'] = (0, params['charge']*10)
            bounds['length'] = (0, maximum_length)
            bounds['width'] = (0, maximum_length)
            bounds['x_cm'] = (self.pix_x.min() * 1.5, self.pix_x.max() * 1.5)
            bounds['y_cm'] = (self.pix_y.min() * 1.5, self.pix_y.max() * 1.5)
            bounds['psi'] = (0, np.pi)
            bounds['t_cm'] = (-readout_max, readout_max)
            bounds['v'] = (-1 / v_min, 1 / v_min)

        else:
            for key, val in params.items():

                if val > 0:

                    bounds[key] = (val * 0.5, val * 2)
                else:

                    bounds[key] = (val * 2, val * 0.5)

            bounds['psi'] = (max(0, bounds['psi'][0]), min(np.pi, bounds['psi'][1]))
            bounds['charge'] = (bounds['charge'][0], bounds['charge'][1] * 2)
            bounds['v'] = (- np.abs(params['v']) * 2, np.abs(params['v']) * 2)

        return bounds

    def plot_ellipse(self, n_sigma=3):

        c, times = self._compute_charge_and_time()
        charge = np.ma.masked_array(np.zeros(self.mask_pixel.shape), mask=~self.mask_pixel)
        charge[self.mask_pixel] = c

        cam_display, _ = plot_array_camera(data=charge, label='$N_{p.e.}$')
        length = n_sigma * self.end_parameters['length']
        psi = self.end_parameters['psi']
        dx = length * np.cos(psi)
        dy = length * np.sin(psi)
        direction_arrow = Arrow(x=self.end_parameters['x_cm'],
                                y=self.end_parameters['y_cm'],
                                dx=dx, dy=dy, width=10, color='k',
                                label='EAS direction')

        cam_display.axes.add_patch(direction_arrow)

        cam_display.add_ellipse(centroid=(self.end_parameters['x_cm'],
                                          self.end_parameters['y_cm']),
                                width=n_sigma * self.end_parameters['width'],
                                length=length,
                                angle=psi,
                                linewidth=6, color='r', linestyle='--',
                                label='{} $\sigma$ contour'.format(n_sigma))
        cam_display.axes.legend(loc='best')

        return cam_display

    def plot_times(self, axes=None):

        c, times = self._compute_charge_and_time()

        dx = (self.pix_x - self.end_parameters['x_cm'])
        dy = (self.pix_y - self.end_parameters['y_cm'])
        long = dx * np.cos(self.end_parameters['psi']) + dy * np.sin(self.end_parameters['psi'])

        x = np.linspace(long.min(), long.max(), num=100)
        y = np.polyval([self.end_parameters['v'], self.end_parameters['t_cm']], x)

        if axes is None:
            fig = plt.figure()
            axes = fig.add_subplot(111)

        axes.scatter(long, times, color='k')
        label = '$t_{CM}=$' + ' {:.2f} [ns]'.format(self.end_parameters['t_cm'])
        label += '$v=$' + '{:.4f} [ns/mm]'.format(self.end_parameters['v'])
        axes.plot(x, y, label=label, color='r')
        axes.legend(loc='best')

        return axes

    def plot_times_camera(self, n_sigma=3):

        x_pix = self.geometry.pix_x.value
        y_pix = self.geometry.pix_y.value
        dx = (x_pix - self.end_parameters['x_cm'])
        dy = (y_pix - self.end_parameters['y_cm'])
        long_pix = dx * np.cos(self.end_parameters['psi']) + dy * np.sin(self.end_parameters['psi'])
        fitted_times = np.polyval([self.end_parameters['v'], self.end_parameters['t_cm']], long_pix)

        fitted_times = np.ma.masked_array(fitted_times,
                                          mask=~self.mask_pixel)
        cam_display, _ = plot_array_camera(data=fitted_times, label='reconstructed time [ns]')
        cam_display.add_ellipse(centroid=(self.end_parameters['x_cm'],
                                          self.end_parameters['y_cm']),
                                width=n_sigma * self.end_parameters['width'],
                                length=n_sigma * self.end_parameters['length'],
                                angle=self.end_parameters['psi'],
                                linewidth=7, color='r')

        return cam_display

    def plot_waveforms(self):

        image = gaussian2d(photo_electrons=self.end_parameters['charge'],
                           x=self.pix_x,
                           y=self.pix_y,
                           x_cm=self.end_parameters['x_cm'],
                           y_cm=self.end_parameters['y_cm'],
                           width=self.end_parameters['width'],
                           length=self.end_parameters['length'],
                           psi=self.end_parameters['psi']) * self.pix_area * self.gain / (1 - self.crosstalk) + self.baseline
        n_pixels = min(15, len(image))
        pixels = np.argsort(image)[-n_pixels:]
        dx = (self.pix_x[pixels] - self.end_parameters['x_cm'])
        dy = (self.pix_y[pixels] - self.end_parameters['y_cm'])
        long_pix = dx * np.cos(self.end_parameters['psi']) + dy * np.sin(self.end_parameters['psi'])
        fitted_times = np.polyval([self.end_parameters['v'], self.end_parameters['t_cm']], long_pix)
        # fitted_times = fitted_times[pixels]
        times_index = np.argsort(fitted_times)

        # waveforms = self.data[times_index]
        waveforms = self.data[pixels]
        waveforms = waveforms[times_index]
        long_pix = long_pix[times_index]
        fitted_times = fitted_times[times_index]

        X, Y = np.meshgrid(self.times, long_pix)

        plt.figure()
        plt.pcolormesh(X, Y, waveforms)
        plt.xlabel('time [ns]')
        plt.ylabel('Longitude [mm]')
        plt.plot(fitted_times, long_pix, color='w', label='Fitted arrival times\n'
                                                          'Velocity : {:.2f} [mm/ns]'.format(1/self.end_parameters['v']))
        l = plt.legend(loc='best',)
        for text in l.get_texts():
            text.set_color("w")
        plt.colorbar(label='[LSB]')

    def plot_waveforms_3D(self):

        image = gaussian2d(photo_electrons=self.end_parameters['charge'],
                           x=self.pix_x,
                           y=self.pix_y,
                           x_cm=self.end_parameters['x_cm'],
                           y_cm=self.end_parameters['y_cm'],
                           width=self.end_parameters['width'],
                           length=self.end_parameters['length'],
                           psi=self.end_parameters[
                               'psi']) * self.pix_area * self.gain / (
                            1 - self.crosstalk) + self.baseline
        n_pixels = min(15, len(image))
        pixels = np.argsort(image)[-n_pixels:]
        image = image[pixels]
        dx = (self.pix_x[pixels] - self.end_parameters['x_cm'])
        dy = (self.pix_y[pixels] - self.end_parameters['y_cm'])
        long_pix = dx * np.cos(self.end_parameters['psi']) + dy * np.sin(
            self.end_parameters['psi'])
        fitted_times = np.polyval(
            [self.end_parameters['v'], self.end_parameters['t_cm']], long_pix)
        # fitted_times = fitted_times[pixels]
        times_index = np.argsort(fitted_times)

        # waveforms = self.data[times_index]
        waveforms = self.data[pixels]
        waveforms = waveforms[times_index]
        long_pix = long_pix[times_index]
        image = image[times_index]
        fitted_times = fitted_times[times_index]
        X, Y = np.meshgrid(self.times, long_pix)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_wireframe(X, Y, waveforms, color='b')

        t = np.linspace(self.times.min(), self.times.max(), num=1000)
        long_fit = np.linspace(long_pix.min(), long_pix.max(), num=1000)
        t_fit = np.polyval([self.end_parameters['v'], self.end_parameters['t_cm']], long_fit)

        for i, waveform in enumerate(waveforms):
            ax.plot(t, long_pix[i] * np.ones(t.shape), self.template(t - fitted_times[i]) * image[i], marker='None', color='r', linestyle='-')
        ax.plot(t_fit, long_fit, 0, color='k')
        ax.view_init(30, 80+180)
        ax.set_xlabel('time [ns]')
        ax.set_ylabel('Longitude [mm]')

        return ax

    def to_container(self, start_parameters=False):

        container = ImageParametersContainer()

        if start_parameters:

            fit_params = self.start_parameters
            fit_errors = {key: np.nan for key in fit_params.keys()}

        else:
            fit_params = self.end_parameters
            fit_errors = self.error_parameters

        container.hillas.intensity = fit_params['charge']
        container.hillas.intensity_err = fit_errors['charge']
        container.hillas.x = fit_params['x_cm']
        container.hillas.x_err = fit_errors['x_cm']
        container.hillas.y = fit_params['y_cm']
        container.hillas.y_err = fit_errors['y_cm']
        container.hillas.r = np.sqrt(
            fit_params['x_cm'] ** 2 + fit_params['y_cm'] ** 2)
        container.hillas.r_err = np.abs(container.hillas.x/container.hillas.r) * container.hillas.x_err + np.abs(container.hillas.y/container.hillas.r) * container.hillas.y_err
        container.hillas.phi = np.arctan(
            fit_params['y_cm'] / fit_params['x_cm'])
        container.hillas.phi_err = np.abs(container.hillas.y / container.hillas.r**2) * container.hillas.x_err + np.abs(container.hillas.x / container.hillas.r**2) * container.hillas.y_err
        container.hillas.length = fit_params['length']
        container.hillas.length_err = fit_errors['length']
        container.hillas.width = fit_params['width']
        container.hillas.width_err = fit_errors['width']
        container.hillas.psi = fit_params['psi']
        container.hillas.psi_err = fit_errors['psi']
        container.hillas.alpha = compute_alpha(container.hillas.phi,
                                               fit_params['psi'])
        container.hillas.alpha_err = fit_errors['psi'] + container.hillas.phi_err
        dx = self.pix_x - fit_params['x_cm']
        dy = self.pix_y - fit_params['y_cm']
        long = dx * np.cos(fit_params['psi']) + dy * np.sin(fit_params['psi'])
        lat = dx * np.sin(fit_params['psi']) - dy * np.cos(fit_params['psi'])

        image = self._compute_charge_and_time()[0]

        if image.sum() > 0:

            skewness_l = np.average(long ** 3, weights=image) / fit_params[
                'length'] ** 3
            kurtosis_l = np.average(long ** 4, weights=image) / fit_params[
                'length'] ** 4

            skewness_w = np.average(lat ** 3, weights=image) / fit_params[
                'width'] ** 3
            kurtosis_w = np.average(lat ** 4, weights=image) / fit_params[
                'width'] ** 4

            container.hillas.skewness_l = skewness_l
            container.hillas.skewness_w = skewness_w
            container.hillas.kurtosis_l = kurtosis_l
            container.hillas.kurtosis_w = kurtosis_w

        if not start_parameters:
            container.log_lh = np.log(self.likelihood(**fit_params))

        container.timing.intercept = fit_params['t_cm']
        container.timing.intercept_err = fit_errors['t_cm']
        container.timing.slope = fit_params['v']
        container.timing.slope_err = fit_errors['v']
        container.hillas.leakage = compute_leakage(x=fit_params['x_cm'],
                                                   y=fit_params['y_cm'],
                                                   psi=fit_params['psi'],
                                                   width=fit_params['width'],
                                                   length=fit_params['length'],
                                                   n_sigma=3,
                                                   geom=self.geometry)

        return container


class SpaceTimeFitter(ImageFitter):

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

    def log_pdf(self, charge, t_cm, x_cm, y_cm, width,
                length, psi, v):

        image = gaussian2d(photo_electrons=charge, x=self.pix_x,
                           y=self.pix_y, x_cm=x_cm, y_cm=y_cm, width=width,
                           length=length, psi=psi)
        image = image * self.gain * self.pix_area / (1 - self.crosstalk)

        dx = (self.pix_x - x_cm)
        dy = (self.pix_y - y_cm)
        long = dx * np.cos(psi) + dy * np.sin(psi)
        p = [v, t_cm]
        t = np.polyval(p, long)
        t = self.times[..., np.newaxis] - t
        t = t.T

        image = image[:, None] * self.template(t) + self.baseline[:, None]

        log_pdf = -((self.data - image) / (np.sqrt(2) * self.error))**2

        log_pdf = log_pdf - np.log(np.sqrt(2*np.pi) * self.error)
        n_points = log_pdf.size
        log_pdf = np.sum(log_pdf) / n_points

        return log_pdf


class PoissonSpaceTimeFitter(ImageFitter):

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

    def log_pdf(self, charge, t_cm, x_cm, y_cm, width,
                length, psi, v):

        image = gaussian2d(photo_electrons=charge, x=self.pix_x, y=self.pix_y,
                           x_cm=x_cm, y_cm=y_cm, width=width,
                           length=length, psi=psi)
        image = image * self.gain * self.pix_area / (1 - self.crosstalk)

        dx = (self.pix_x - x_cm)
        dy = (self.pix_y - y_cm)
        long = dx * np.cos(psi) + dy * np.sin(psi)
        p = [v, t_cm]
        t = np.polyval(p, long)
        t = self.times[..., np.newaxis] - t
        t = t.T
        image = image[:, None] * self.template(t) + self.baseline[:, None]

        mask = (self.data > 0) * (image > 0) * (np.isfinite(image))
        image = image[mask]
        data = self.data[mask]

        log_pdf = np.log(image) * data - image - gammaln(data + 1)
        n_points = log_pdf.size
        log_pdf = np.sum(log_pdf) / n_points

        return log_pdf


class HillasFitter(ImageFitter):

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

        self.data_integrated, _ = self._compute_charge_and_time()
        #self.data_integrated = np.delete(self.data_integrated, self.mask_pixel,
        #                                 axis=0)
        self.error_integrated = np.sqrt(self.integral_width) * self.error.mean(axis=-1)
        #self.error_integrated = np.delete(self.error_integrated, self.mask_pixel, axis=0)

    def log_pdf(self, charge, t_cm, x_cm, y_cm, width,
                length, psi, v):

        image = gaussian2d(photo_electrons=charge, x=self.pix_x, y=self.pix_y,
                           x_cm=x_cm, y_cm=y_cm, width=width,
                           length=length, psi=psi)
        image = image * self.pix_area

        log_pdf = -((self.data_integrated - image) / (np.sqrt(2) * self.error_integrated)) ** 2

        log_pdf = log_pdf - np.log(np.sqrt(2 * np.pi) * self.error_integrated)
        n_points = log_pdf.size
        log_pdf = np.sum(log_pdf) / n_points

        return log_pdf

    def fit(self, verbose=True, **kwargs):

        super().fit(verbose=verbose, **kwargs,
                    t_cm=self.start_parameters['t_cm'],
                    v=self.start_parameters['v'])


class SPESpaceTimeFitter(ImageFitter):


    def __init__(self, *args, n_peaks=100, **kwargs):

        super().__init__(*args, **kwargs)

        self._initialize_pdf(n_peaks=n_peaks)

    def _initialize_pdf(self, n_peaks):
        photoelectron_peak = np.arange(n_peaks, dtype=np.int)
        self.photo_peaks = photoelectron_peak
        photoelectron_peak = photoelectron_peak[..., None]
        sigma_n = self.error[:, 0] ** 2 + photoelectron_peak * self.sigma_s ** 2
        sigma_n = np.sqrt(sigma_n)
        self.sigma_n = sigma_n

        self.photo_peaks
        mask = (self.photo_peaks == 0)
        self.photo_peaks[mask] = 1
        log_k = np.log(self.photo_peaks)
        log_k = np.cumsum(log_k)
        self.photo_peaks[mask] = 0
        self.log_k = log_k
        self.crosstalk_factor = photoelectron_peak * self.crosstalk
        self.crosstalk_factor = self.crosstalk_factor

    
    def log_pdf(self, charge, t_cm, x_cm, y_cm, width,
                length, psi, v):
        dx = (self.pix_x - x_cm)
        dy = (self.pix_y - y_cm)
        long = dx * np.cos(psi) + dy * np.sin(psi)
        p = [v, t_cm]
        t = np.polyval(p, long)
        t = self.times[..., np.newaxis] - t
        t = t.T

        ## mu = mu * self.pix_area

        # mu = mu[..., None] * self.template(t)
        # mask = (mu > 0)
        log_mu = log_gaussian2d(size=charge * self.pix_area,
                                x=self.pix_x,
                                y=self.pix_y,
                                x_cm=x_cm,
                                y_cm=y_cm,
                                width=width,
                                length=length,
                                psi=psi)
        mu = np.exp(log_mu)

        # log_mu[~mask] = -np.inf
        log_k = self.log_k

        x = mu + self.crosstalk_factor
        # x = np.rollaxis(x, 0, 3)
        log_x = np.log(x)
        # mask = x > 0
        # log_x[~mask] = -np.inf

        log_x = ((self.photo_peaks - 1) * log_x.T).T
        log_poisson = log_mu - log_k[..., None] - x + log_x
        # print(log_poisson)

        mean = self.photo_peaks * ((self.gain[..., None] * self.template(t)))[
            ..., None]
        x = self.data - self.baseline[..., None]
        sigma_n = np.expand_dims(self.sigma_n.T, axis=1)

        log_gauss = log_gaussian(x[..., None], mean, sigma_n)

        log_poisson = np.expand_dims(log_poisson.T, axis=1)
        log_pdf = log_poisson + log_gauss
        pdf = np.sum(np.exp(log_pdf), axis=-1)

        mask = (pdf <= 0)
        pdf = pdf[~mask]
        n_points = pdf.size
        log_pdf = np.log(pdf).sum() / n_points

        return log_pdf
    """
    def log_pdf(self, charge, t_cm, x_cm, y_cm, width,
                length, psi, v):

        n_pixels, n_samples = self.data.shape
        # print(self.data.dtype, self.pix_x.dtype, self.error.dtype)


        y = image_loglikelihood(self.data.ravel().astype(np.float64),
                            self.pix_x,
                            self.pix_y,
                            self.pix_area,
                            n_pixels,
                            n_samples,
                            self.times[0],
                            self.dt,
                            self.error[:, 0],
                            self.sigma_s,
                            self.crosstalk,
                            self.gain,
                            self.baseline,
                            self.template._template._spline.t,
                            self.template._template._spline.c,
                            self.template._template._spline.k,
                            len(self.template._template._spline.t),
                            t_cm,
                            v,
                            x_cm,
                            y_cm,
                            psi,
                            width,
                            length,
                            charge)

        return y
    """

class ShowerFitter(Fitter):

    def __init__(self, data, gain=np.ones(1296), baseline=np.zeros(1296),
                 crosstalk=np.ones(1296) * 0.08,
                 template=NormalizedPulseTemplate.load(TEMPLATE_FILENAME),
                 error=None,
                 geometry=GEOMETRY,):
        dt = 4
        self.integral_width = 7
        self.n_pixels, self.n_samples = data.shape
        self.times = np.arange(0, self.n_samples) * dt
        self.template = template
        self.template_time_max = template.compute_time_of_max()
        self.template_gain_factor = template.compute_charge_amplitude_ratio(self.integral_width, dt)
        self.error = error if error is not None else np.ones(data.shape)
        self.geometry = geometry
        self.pix_x = geometry.pix_x.value
        self.pix_y = geometry.pix_y.value
        self.pix_area = geometry.pix_area
        self.gain = gain
        self.baseline = baseline
        self.crosstalk = crosstalk
        self.labels = {'charge': 'Charge [LSB]',
                       't_cm': '$t_{CM}$ [ns]',
                       'x_cm': '$x_{CM}$ [m]',
                       'y_cm': '$y_{CM}$ [m]',
                       'width': '$\sigma_w$ [m]',
                       'length': '$\sigma_l$ [m]',
                        'psi': '$\psi$ [rad]',
                        'v': '$v$ [m/ns]'
        }


        super().__init__(data)

        self.mask_pixel, self.mask_time = self.clean_data(sigma_space=4, sigma_time=3)
        pixels = np.arange(self.n_pixels)[~self.mask_pixel]
        t = np.arange(self.n_samples)[~self.mask_time]
        self.data = np.delete(self.data, pixels, axis=0)
        self.data = np.delete(self.data, t, axis=1)
        self.error = np.delete(self.error, pixels, axis=0)
        self.error = np.delete(self.error, t, axis=1)
        self.times = self.times[self.mask_time]
        self.gain = self.gain[self.mask_pixel]
        self.crosstalk = self.crosstalk[self.mask_pixel]
        self.baseline = self.baseline[self.mask_pixel]
        self.pix_x = self.pix_x[self.mask_pixel]
        self.pix_y = self.pix_y[self.mask_pixel]
        self.pix_area = self.pix_area[self.mask_pixel]

    def log_pdf(self, charge, t_cm, x_cm, y_cm, width,
                length, psi, v):

        n_points = self.data.shape[0] * self.data.shape[1]
        dx = (self.pix_x - x_cm)
        dy = (self.pix_y - y_cm)
        long = dx * np.cos(psi) + dy * np.sin(psi)
        p = [v, t_cm]
        t = np.polyval(p, long)

        A = gaussian2d(photo_electrons=1,
                         x=self.pix_x, y=self.pix_y, x_cm=x_cm,
                         y_cm=y_cm, width=width, length=length,
                         psi=psi)
        A = A / self.gain * self.pix_area / (1 - self.crosstalk)
        t = self.times[..., np.newaxis] - t
        t = t.T
        y_fit = charge * A[..., np.newaxis] * self.template(t) + self.baseline[..., np.newaxis]
        norm = np.sqrt(1 / (2 * np.pi)) / self.error
        log_lh = - ((y_fit - self.data) / self.error)**2 * 0.5 + np.log(norm)
        log_lh = np.sum(log_lh, axis=-1) / n_points
        return log_lh

    def compute_chargeand_time(self):



        w = np.array([1, 2, 3, 4, 5, 4, 3, 2, 1], dtype=np.float32)
        w /= w.sum()
        integral_width = self.integral_width
        pulse_mask = np.zeros(self.data.shape, dtype=np.bool)
        threshold = 7.5 * 3
        # threshold = np.std(self.data.ravel(), axis=-1) * 2
        # print(threshold)
        # threshold = threshold[:, None]
        # threshold = self.error[:, 1:-1] * 5
        c = convolve1d(
            input=self.data,
            weights=w,
            axis=1,
            mode='constant',
        )

        pulse_mask[:, 1:-1] = (
                (c[:, :-2] <= c[:, 1:-1]) &
                (c[:, 1:-1] >= c[:, 2:]) &
                (c[:, 1:-1] > threshold)
        )

        times = np.argmax(pulse_mask, axis=-1)
        times = self.times[times] - self.template_time_max

        charge = convolve1d(
            self.data,
            np.ones(integral_width),
            axis=-1
        )
        charge = (charge * pulse_mask).sum(axis=-1)
        charge = charge * self.template.compute_charge_amplitude_ratio(
            integral_width=integral_width, dt_sampling=4)
        """
        times = np.argmax(self.data, axis=-1)
        charge = np.max(self.data, axis=-1)
        times = self.times[times] - self.template_time_max
        """

        charge = (charge - self.baseline) / (self.gain) * (1 - self.crosstalk)
        charge[charge < 0] = 0

        return charge, times

    def clean_image(self, charge, times):


        mask = cleaning.tailcuts_clean(geom=self.geometry,
                                       image=charge,
                                       picture_thresh=15,
                                       boundary_thresh=10, )
        # plot_array_camera(mask.astype(float))

        # mask = apply_time_delta_cleaning(self.geometry, mask, times,
        #                      min_number_neighbors=3,
        #                          time_limit=10)
        # plot_array_camera(mask.astype(float))
        # plt.show()

        return mask

    def initialize_fit(self):

        charge, times = self.compute_chargeand_time()
        mask = self.clean_image(charge, times)
        # print('Initial charge', np.sum(charge))

        charge[~mask] = 0
        # charge = np.ma.masked_array(charge, mask=mask)
        # print(charge.sum())
        times = np.ma.masked_array(times, mask=mask)
        hillas = hillas_parameters(self.geometry, charge)
        timing = timing_parameters(self.geometry, charge,
                                   times, hillas)

        # plot_array_camera(charge)
        # plt.show()


        size = hillas.intensity
        t_cm = timing.intercept
        x_cm = hillas.x.value
        y_cm = hillas.y.value
        width = hillas.width.value
        length = hillas.length.value
        psi = hillas.psi.value
        v = timing.slope.value

        params = [size, t_cm, x_cm, y_cm, width, length, psi, v]
        params = dict(zip(self.names_parameters, params))

        # print(self.data.shape)
        # 0/0

        return params

    def clean_data(self, sigma_space=5, sigma_time=3):

        x_cm = self.start_parameters['x_cm']
        y_cm = self.start_parameters['y_cm']
        width = self.start_parameters['width']
        length = self.start_parameters['length']
        psi = self.start_parameters['psi']

        dx = self.pix_x - x_cm
        dy = self.pix_y - y_cm

        lon = dx * np.cos(psi) + dy * np.sin(psi)
        lat = dx * np.sin(psi) - dy * np.cos(psi)

        mask_pixel = ((lon / length)**2 + (lat / width)**2) < sigma_space**2

        v = self.start_parameters['v']
        t_start = self.start_parameters['t_cm'] - (np.abs(v) * length/2 * sigma_time) - 20
        t_end = self.start_parameters['t_cm'] + (np.abs(v) * length/2 * sigma_time) + 20

        mask_time = (self.times < t_end) * (self.times > t_start)

        return mask_pixel, mask_time

    def compute_bounds(self, method='limit'):

        params = self.start_parameters
        if method == 'limit':

            maximum_length = 5000
            readout_max = 200
            v_min = 1/200 # 1 mm / per ns
            bounds = {}
            bounds['charge'] = (0, np.inf)
            bounds['length'] = (0, maximum_length)
            bounds['width'] = (0, maximum_length)
            bounds['x_cm'] = (self.pix_x.min()*1.5, self.pix_x.max()*1.5)
            bounds['y_cm'] = (self.pix_y.min()*1.5, self.pix_y.max()*1.5)
            bounds['psi'] = (0, 2 * np.pi)
            bounds['t_cm'] = (-readout_max, readout_max)
            bounds['v'] = (-1/v_min, 1/v_min)

        else:
            bounds = {}

            for key, val in params.items():

                if val > 0:

                    bounds[key] = (val * 0.5, val * 2)
                else:

                    bounds[key] = (val * 2, val * 0.5)
            # bounds['psi'] = (max(0, bounds['psi'][0]), min(np.pi, bounds['psi'][1]))
            bounds['charge'] = (bounds['charge'][0], bounds['charge'][1] * 2)

        return bounds

    def plot(self, n_sigma=3, **kwargs):

        c, times = self.compute_chargeand_time()
        charge = np.ma.masked_array(np.zeros(self.mask_pixel.shape), mask=~self.mask_pixel)
        # times = np.zeros(self.mask_pixel.shape) * np.nan
        charge[self.mask_pixel] = c
        # times[self.mask_pixel] = t
        # times = times - self.end_parameters['t_cm']

        cam_display, _ = plot_array_camera(data=charge)
        cam_display.add_ellipse(centroid=(self.end_parameters['x_cm'],
                                          self.end_parameters['y_cm']),
                                width=n_sigma * self.end_parameters['width'],
                                length=n_sigma * self.end_parameters['length'],
                                angle=self.end_parameters['psi'],
                                linewidth=7, color='r')

        return cam_display

    def plot_times(self, n_sigma=3):

        c, times = self.compute_chargeand_time()

        dx = (self.pix_x - self.end_parameters['x_cm'])
        dy = (self.pix_y - self.end_parameters['y_cm'])
        long = dx * np.cos(self.end_parameters['psi']) + dy * np.sin(self.end_parameters['psi'])

        x = np.linspace(long.min(), long.max(), num=100)
        y = np.polyval([self.end_parameters['v'], self.end_parameters['t_cm']], x)
        plt.figure()
        plt.scatter(long, times, color='k')
        plt.plot(x, y, label='Fit', color='r')

        x_pix = self.geometry.pix_x.value
        y_pix = self.geometry.pix_y.value
        dx = (x_pix - self.end_parameters['x_cm'])
        dy = (y_pix - self.end_parameters['y_cm'])
        long_pix = dx * np.cos(self.end_parameters['psi']) + dy * np.sin(self.end_parameters['psi'])
        fitted_times = np.polyval([self.end_parameters['v'], self.end_parameters['t_cm']], long_pix)

        fitted_times = np.ma.masked_array(fitted_times,
                                          mask=~self.mask_pixel)
        cam_display, _ = plot_array_camera(data=fitted_times)
        cam_display.add_ellipse(centroid=(self.end_parameters['x_cm'],
                                          self.end_parameters['y_cm']),
                                width=n_sigma * self.end_parameters['width'],
                                length=n_sigma * self.end_parameters['length'],
                                angle=self.end_parameters['psi'],
                                linewidth=7, color='r')

        return cam_display

    def plot_waveforms(self):

        image = gaussian2d(photo_electrons=self.end_parameters['charge'],
                           x=self.pix_x,
                           y=self.pix_y,
                           x_cm=self.end_parameters['x_cm'],
                           y_cm=self.end_parameters['y_cm'],
                           width=self.end_parameters['width'],
                           length=self.end_parameters['length'],
                           psi=self.end_parameters['psi']) * self.pix_area * self.gain / (1 - self.crosstalk) + self.baseline
        n_pixels = min(15, len(image))
        pixels = np.argsort(image)[-n_pixels:]
        image = image[pixels]
        dx = (self.pix_x[pixels] - self.end_parameters['x_cm'])
        dy = (self.pix_y[pixels] - self.end_parameters['y_cm'])
        long_pix = dx * np.cos(self.end_parameters['psi']) + dy * np.sin(self.end_parameters['psi'])
        fitted_times = np.polyval([self.end_parameters['v'], self.end_parameters['t_cm']], long_pix)
        # fitted_times = fitted_times[pixels]
        times_index = np.argsort(fitted_times)

        # waveforms = self.data[times_index]
        waveforms = self.data[pixels]
        waveforms = waveforms[times_index]
        long_pix = long_pix[times_index]
        image = image[times_index]
        fitted_times = fitted_times[times_index]

        X, Y = np.meshgrid(self.times, long_pix)

        plt.figure()
        plt.pcolormesh(X, Y, waveforms)
        plt.xlabel('time [ns]')
        plt.ylabel('Longitude [m]')
        plt.plot(fitted_times, long_pix, color='k', label='Fitted arrival times\n'
                                                          'Velocity : {} [ns/mm]'.format(self.end_parameters['v']))
        plt.legend(loc='best')
        plt.colorbar(label='[LSB]')

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_wireframe(X, Y, waveforms, color='b')

        t = np.linspace(self.times.min(), self.times.max(), num=1000)
        long_fit = np.linspace(long_pix.min(), long_pix.max(), num=1000)
        t_fit = np.polyval([self.end_parameters['v'], self.end_parameters['t_cm']], long_fit)

        for i, waveform in enumerate(waveforms):
            ax.plot(t, long_pix[i] * np.ones(t.shape), self.template(t - fitted_times[i]) * image[i], marker='None', color='r', linestyle='-')
        ax.plot(t_fit, long_fit, 0, color='k')
        ax.view_init(30, 80+180)
        ax.set_xlabel('time [ns]')
        ax.set_ylabel('Longitude [m]')


class MPEShowerFitter(ShowerFitter):

    def __init__(self, data,
                 gain,
                 crosstalk,
                 baseline,
                 sigma_e=np.ones(1296),
                 sigma_s=np.ones(1296),
                 template=NormalizedPulseTemplate.load(TEMPLATE_FILENAME),
                 error=None,
                 geometry=GEOMETRY, n_peaks=1000):

        self.sigma_e = sigma_e
        self.sigma_s = sigma_s

        super().__init__(data=data,
                         gain=gain,
                         crosstalk=crosstalk,
                         baseline=baseline,
                 template=template,
                 error=error,
                 geometry=geometry, )

        sigma_e = np.std(data, axis=-1)
        sigma_e = np.mean(sigma_e[~self.mask_pixel])

        self.sigma_e = np.ones(self.mask_pixel.sum()) * sigma_e
        self.sigma_s = self.sigma_s[self.mask_pixel]

        self._initialize_pdf(n_peaks=n_peaks)

    def _initialize_pdf(self, n_peaks):

        photoelectron_peak = np.arange(n_peaks, dtype=np.int)
        self.photo_peaks = photoelectron_peak
        photoelectron_peak = photoelectron_peak[..., None]
        sigma_n = self.sigma_e ** 2 + photoelectron_peak * self.sigma_s ** 2
        sigma_n = np.sqrt(sigma_n)
        self.sigma_n = sigma_n

        self.photo_peaks
        mask = (self.photo_peaks == 0)
        self.photo_peaks[mask] = 1
        log_k = np.log(self.photo_peaks)
        log_k = np.cumsum(log_k)
        self.photo_peaks[mask] = 0
        self.log_k = log_k
        self.crosstalk_factor = photoelectron_peak * self.crosstalk
        self.crosstalk_factor = self.crosstalk_factor

    def log_pdf(self, charge, t_cm, x_cm, y_cm, width,
                length, psi, v):


        dx = (self.pix_x - x_cm)
        dy = (self.pix_y - y_cm)
        long = dx * np.cos(psi) + dy * np.sin(psi)
        p = [v, t_cm]
        t = np.polyval(p, long)
        t = self.times[..., np.newaxis] - t
        t = t.T

        ## mu = mu * self.pix_area

        # mu = mu[..., None] * self.template(t)
        # mask = (mu > 0)
        log_mu = log_gaussian2d(size=charge*self.pix_area,
                        x=self.pix_x,
                        y=self.pix_y,
                        x_cm=x_cm,
                        y_cm=y_cm,
                        width=width,
                        length=length,
                        psi=psi)
        mu = np.exp(log_mu)

        # log_mu[~mask] = -np.inf
        log_k = self.log_k

        x = mu + self.crosstalk_factor
        # x = np.rollaxis(x, 0, 3)
        log_x = np.log(x)
        # mask = x > 0
        # log_x[~mask] = -np.inf

        log_x = ((self.photo_peaks - 1) * log_x.T).T
        log_poisson = log_mu - log_k[..., None] - x + log_x
        # print(log_poisson)

        mean = self.photo_peaks * ((self.gain[..., None] * self.template(t)))[..., None]
        x = self.data - self.baseline[..., None]
        sigma_n = np.expand_dims(self.sigma_n.T, axis=1)

        log_gauss = log_gaussian(x[..., None], mean, sigma_n)

        log_poisson = np.expand_dims(log_poisson.T, axis=1)
        log_pdf = log_poisson + log_gauss
        pdf = np.sum(np.exp(log_pdf), axis=-1)
        mask = (pdf <= 0)
        pdf = pdf[~mask]
        log_pdf = np.log(pdf)

        return log_pdf