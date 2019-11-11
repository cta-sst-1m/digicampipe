from abc import abstractmethod, ABC
import numpy as np
from scipy.optimize import minimize
from pkg_resources import resource_filename
import os
import inspect
from digicampipe.utils.pulse_template import NormalizedPulseTemplate
import matplotlib.pyplot as plt
from copy import copy
from digicampipe.instrument.camera import DigiCam
from digicampipe.utils.pdf import gaussian2d
from scipy.ndimage import convolve1d
from ctapipe.image import cleaning, hillas_parameters
from ctapipe.image.cleaning import apply_time_delta_cleaning
from ctapipe.image.timing_parameters import timing_parameters
from digicampipe.visualization.plot import plot_array_camera

GEOMETRY = DigiCam.geometry

TEMPLATE_FILENAME = resource_filename(
    'digicampipe',
    os.path.join(
        'tests',
        'resources',
        'pulse_SST-1M_pixel_0.dat'
    )
)


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
        str += 'Likelihood :\t{}'.format(self.likelihood(**self.end_parameters))

        return str

    def fit(self, verbose=True, **kwargs):

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
                                         np.diagonal(self.correlation_matrix)))
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

        x = np.linspace(self.end_parameters[key] * 0.5, self.end_parameters[key] * 1.5, num=size)
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
            axes.axvspan(self.bounds[key][0],
                         self.bounds[key][1], label='bounds',
                         alpha=0.5, facecolor='k')
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
        x = np.linspace(self.end_parameters[key_x]*0.5, self.end_parameters[key_x] * 1.5, num=size[0])
        y = np.linspace(self.end_parameters[key_y]*0.5, self.end_parameters[key_y] * 1.5, num=size[1])
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
        t_0 = self.times[index_max] - 10
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


class ShowerFitter(Fitter):

    def __init__(self, data,
                 template=NormalizedPulseTemplate.load(TEMPLATE_FILENAME),
                 error=None,
                 geometry=GEOMETRY,
                 gain=None):

        self.n_pixels, self.n_samples = data.shape
        self.times = np.arange(0, self.n_samples) * 4
        self.template = template
        self.error = error if error is not None else np.ones(self.n_samples)
        self.geometry = geometry
        self.pix_x = geometry.pix_x.value
        self.pix_y = geometry.pix_y.value
        self.gain = np.ones(self.n_pixels) * 20
        self.baseline = np.zeros(self.n_pixels)
        self.crosstalk = np.ones(self.n_pixels) * 0.08
        self.labels = {'charge': 'Charge [LSB]',
                       't_cm': '$t_{CM}$ [ns]',
                       'x_cm': '$x_{CM}$ [mm]',
                       'y_cm': '$y_{CM}$ [mm]',
                       'width': '$\sigma_w$ [mm]',
                       'length': '$\sigma_l$ [mm]',
                        'psi': '$\psi$ [rad]',
                        'v': '$v$ [mm/ns]'
        }
        super().__init__(data)

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
        integral_width = 7
        pulse_mask = np.zeros(self.data.shape, dtype=np.bool)
        threshold = self.error[1:-1] * 5
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
        times = self.times[times]

        charge = convolve1d(
            self.data,
            np.ones(integral_width),
            axis=-1
        )
        charge = (charge * pulse_mask).sum(axis=-1)

        charge = (charge - self.baseline) / self.gain * (1 - self.crosstalk)
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

        charge[~mask] = 0
        times = np.ma.masked_array(times, mask=mask)
        hillas = hillas_parameters(self.geometry, charge)
        timing = timing_parameters(self.geometry, charge,
                                   times, hillas)

        # plot_array_camera(charge)
        # plt.show()

        charge = hillas.intensity
        t_cm = timing.intercept
        x_cm = hillas.x.value
        y_cm = hillas.y.value
        width = hillas.width.value
        length = hillas.length.value
        psi = hillas.psi.value
        v = timing.slope.value

        params = [charge, t_cm, x_cm, y_cm, width, length, psi, v]
        params = dict(zip(self.names_parameters, params))

        return params

    def compute_bounds(self):

        params = self.start_parameters

        bounds = {}

        for key, val in params.items():

            if val > 0:

                bounds[key] = (val * 0.5, val * 1.5)
            else:

                bounds[key] = (val * 1.5, val * 0.5)
        bounds['psi'] = (0, 2 * np.pi)
        bounds['charge'] = (0, bounds['charge'][1])

        return bounds

    def plot(self, n_sigma=3, **kwargs):

        charge, times = self.compute_chargeand_time()

        mask = cleaning.tailcuts_clean(geom=self.geometry,
                                       image=charge,
                                       picture_thresh=15,
                                       boundary_thresh=10, )

        cam_display, _ = plot_array_camera(data=charge)
        cam_display.add_ellipse(centroid=(self.end_parameters['x_cm'],
                                          self.end_parameters['y_cm']),
                                width=n_sigma * self.end_parameters['width'],
                                length=n_sigma * self.end_parameters['length'],
                                angle=self.end_parameters['psi'],
                                linewidth=7, color='r')

        dx = (self.pix_x - self.end_parameters['x_cm'])
        dy = (self.pix_y - self.end_parameters['y_cm'])
        long = dx * np.cos(self.end_parameters['psi']) + dy * np.sin(self.end_parameters['psi'])

        x = np.linspace(long.min(), long.max(), num=100)
        y = np.polyval([self.end_parameters['v'], self.end_parameters['t_cm']], x)
        plt.figure()
        plt.scatter(long[mask], times[mask], color='k')
        plt.plot(x, y, label='Fit', color='r')

        return cam_display

