import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt


class OpticalElement:

    _wavelength_reference = np.arange(100., 1200., 1.)

    def __init__(self, wavelength, efficiency, kind='linear', **kwargs):

        self._wavelength = wavelength
        self._efficiency = efficiency

        assert np.all((efficiency >= 0.) * (efficiency <= 1.))

        order = np.argsort(self._wavelength)
        self._wavelength = self._wavelength[order]
        self._efficiency = self._efficiency[order]

        self._func = interp1d(self._wavelength,
                              self._efficiency, kind=kind,
                              fill_value=0,
                              bounds_error=False,
                              **kwargs)
        #y = self(self._wavelength_reference)
        #y_max = y.max()
        #y /= y_max
        #self._efficiency /= y_max
        #self._func = interp1d(self._wavelength_reference,
        #                      y, kind=kind,
        #                      fill_value=0,
        #                      bounds_error=False,
        #                      **kwargs)

    def __call__(self, x, **kwargs):

        return self._func(x)

    def __add__(self, other):

        x = self._wavelength_reference
        y_self = self(x)
        y_other = other(x)
        y = y_self * y_other

        return OpticalElement(x, y)

    def plot(self, axes=None, show_datapoints=False, **kwargs):

        if axes is None:

            fig = plt.figure()
            axes = fig.add_subplot(111)

        if show_datapoints:

            axes.scatter(self._wavelength, self._efficiency)
        x = self._wavelength_reference
        y = self(x)
        axes.plot(x, y, **kwargs)
        axes.set_xlabel('Wavelength [nm]')
        axes.set_ylabel('Efficiency [a.u.]')
        axes.legend(loc='best')

        return axes

    def total_efficiency(self, light_source):

        x = self._wavelength_reference
        efficiency = (self(x) * light_source(x)).sum()
        efficiency /= light_source(x).sum()

        return efficiency

    @classmethod
    def load(cls, filename, **kwargs):

        wavelength, efficiency = np.loadtxt(filename).T[0:2]

        return cls(wavelength=wavelength, efficiency=efficiency, **kwargs)


class LightSource(OpticalElement):

    def __init__(self, wavelength, efficiency, kind='linear', **kwargs):

        super().__init__(wavelength, efficiency, kind='linear', **kwargs)
        y = self(self._wavelength_reference)
        y_max = y.max()
        y /= y_max
        self._efficiency /= y_max
        self._func = interp1d(self._wavelength_reference,
                             y, kind=kind,
                             fill_value=0,
                             bounds_error=False,
                             **kwargs)


if __name__ == '__main__':

    file_CTS_led = '/home/alispach/Documents/PhD/ctasoft/digicam-extra/' \
                   'digicamextra/lightsource/cts_led_spectrum.txt'

    file_flasher_led = '/home/alispach/Documents/PhD/ctasoft/digicam-extra/' \
                   'digicamextra/lightsource/flasher_led_spectrum.txt'

    file_window = '/home/alispach/Documents/PhD/ctasoft/digicam-extra/' \
                  'digicamextra/window/transmittance_averaged_first_window.dat'

    file_window_2 = '/home/alispach/Documents/PhD/ctasoft/digicam-extra/' \
                  'digicamextra/window/transmittance_averaged_second_window.dat'

    file_PDE = '/home/alispach/Documents/PhD/ctasoft/digicam-extra/digicamextra/photosensor/PDEData_2.8V_SiPM_LCT2_5477_0000096_DEC2016_reduced.txt'

    file_cherenkov = '/home/alispach/Documents/PhD/ctasoft/digicam-extra/digicamextra/lightsource/cherenkov_spectrum_CTA.dat'
    file_nsb = '/home/alispach/Documents/PhD/ctasoft/digicam-extra/digicamextra/lightsource/nsb_reference_CTA.dat'

    CTS_LED = LightSource.load(file_CTS_led)
    cherenkov_spectrum = LightSource.load(file_cherenkov)
    mask = (cherenkov_spectrum._wavelength_reference >= 300) * (cherenkov_spectrum._wavelength_reference <= 550)
    w = cherenkov_spectrum._wavelength_reference[mask]
    eff = cherenkov_spectrum(w)
    cherenkov_spectrum_cut = LightSource(w, eff)

    flasher_led = LightSource.load(file_flasher_led)

    nsb_spectrum = LightSource.load(file_nsb)
    window = OpticalElement.load(file_window)
    window_2 = OpticalElement.load(file_window_2)
    sipm = OpticalElement.load(file_PDE)

    wavelength = OpticalElement._wavelength_reference
    efficiency = 0.81 * np.ones(wavelength.shape)

    cones = OpticalElement(efficiency=efficiency, wavelength=wavelength)

    camera = sipm + window + cones
    camera_no_window = sipm + cones
    figsize = (12, 10)
    fig = plt.figure(figsize=figsize)
    axes = fig.add_subplot(111)
    CTS_LED.plot(axes=axes, label='CTS LED')
    cherenkov_spectrum.plot(axes=axes, label='Cherenkov Spectrum')
    # cherenkov_spectrum_cut.plot(axes=axes, label='Cherenkov Spectrum (300 - 550 nm)')

    # nsb_spectrum.plot(axes=axes, label='NSB Spectrum')
    flasher_led.plot(axes=axes, label='Flasher LED')
    axes.set_ylabel('Density probability [a.u.]')
    axes.set_xlim(260, 1000)

    fig = plt.figure(figsize=figsize)
    axes = fig.add_subplot(111)
    window.plot(axes=axes, label='Window')
    sipm.plot(axes=axes, label='SiPM')
    cones.plot(axes=axes, label='Cones')
    camera.plot(axes=axes, label='Camera' )
    axes.set_ylabel('Optical efficiency $\eta$ []')
    axes.set_xlim(260, 1000)

    print('CTS LED')
    print(window.total_efficiency(CTS_LED))
    print(sipm.total_efficiency(CTS_LED))
    print(cones.total_efficiency(CTS_LED))
    print(camera.total_efficiency(CTS_LED))
    print(camera_no_window.total_efficiency(CTS_LED))

    print('Cherenkov')
    print(window.total_efficiency(cherenkov_spectrum))
    print(sipm.total_efficiency(cherenkov_spectrum))
    print(cones.total_efficiency(cherenkov_spectrum))
    print(camera.total_efficiency(cherenkov_spectrum))
    print(camera_no_window.total_efficiency(cherenkov_spectrum))

    print('Cherenkov (300-550 nm)')
    print(window.total_efficiency(cherenkov_spectrum_cut))
    print(sipm.total_efficiency(cherenkov_spectrum_cut))
    print(cones.total_efficiency(cherenkov_spectrum_cut))
    print(camera.total_efficiency(cherenkov_spectrum_cut))
    print(camera_no_window.total_efficiency(cherenkov_spectrum_cut))

    print('Flasher LED')
    print(window.total_efficiency(flasher_led))
    print(sipm.total_efficiency(flasher_led))
    print(cones.total_efficiency(flasher_led))
    print(camera.total_efficiency(flasher_led))
    print(camera_no_window.total_efficiency(flasher_led))

    plt.show()
