"""
Display the NSB rate for each pixel
Usage:
  nsb_rate_camera.py [options] [--] <INPUT>

Options:
  --help                        Show this
  <INPUT>                       File of histogram of baselines during a run.
                                Output of raw.py with --baseline_filename on
                                science data.
  --dark_hist=LIST              Histogram of ADC samples during dark run.
                                Output of raw.py on dark data.
  --plot=FILE                   path to the output plot. Will show the average
                                over all events of the NSB.
                                If set to "show", the plot is displayed and not
                                saved.
                                If set to "none", no plot is done.
                                [Default: show]
  --parameters=FILE             Calibration parameters file path
  --template=FILE               Pulse template file path
  --bias_resistance=FLOAT       Bias resistance in Ohm. [Default: 1e4]
  --cell_capacitance=FLOAT      Cell capacitance in Farad. [Default: 5e-14]
"""
import os
import matplotlib.pyplot as plt
import numpy as np
from ctapipe.visualization import CameraDisplay
from docopt import docopt
import yaml
from astropy import units as u

from histogram.histogram import Histogram1D
from digicampipe.calib.baseline import _compute_nsb_rate
from digicampipe.instrument.camera import DigiCam
from digicampipe.utils.docopt import convert_text, convert_float
from digicampipe.utils.pulse_template import NormalizedPulseTemplate
from digicampipe.scripts.bad_pixels import get_bad_pixels
from digicampipe.calib.charge import _get_average_matrix_bad_pixels


def nsb_rate(
        baseline_histo_file, dark_histo_file, param_file, template_filename,
        plot="show", plot_nsb_range=None, norm="log",
        bias_resistance=1e4 * u.Ohm, cell_capacitance=5e-14 * u.Farad
):
    baseline_histo = Histogram1D.load(baseline_histo_file)
    dark_histo = Histogram1D.load(dark_histo_file)
    baseline_shift = baseline_histo.mean()-dark_histo.mean()
    n_pixel = len(DigiCam.geometry.neighbors)
    pixels = np.arange(n_pixel, dtype=int)
    with open(param_file) as file:
        pulse_template = NormalizedPulseTemplate.load(template_filename)
        pulse_area = pulse_template.integral() * u.ns
        charge_to_amplitude = pulse_template.compute_charge_amplitude_ratio(7, 4)
        calibration_parameters = yaml.load(file)
        gain_integral = np.array(calibration_parameters['gain'])
        gain_amplitude = gain_integral * charge_to_amplitude
        crosstalk = np.array(calibration_parameters['mu_xt'])
    rate = _compute_nsb_rate(
        baseline_shift=baseline_shift, gain=gain_amplitude,
        pulse_area=pulse_area, crosstalk=crosstalk,
        bias_resistance=bias_resistance, cell_capacitance=cell_capacitance
    )
    bad_pixels = get_bad_pixels(
        calib_file=param_file, nsigma_gain=5, nsigma_elecnoise=5,
        dark_histo=dark_histo_file, nsigma_dark=8, plot=None, output=None
    )
    bad_pixels = np.unique(np.hstack(
        (
            bad_pixels,
            pixels[rate < 0],
            pixels[rate > 5 * u.GHz]
        )
    ))
    avg_matrix = _get_average_matrix_bad_pixels(DigiCam.geometry, bad_pixels)
    good_pixels_mask = np.ones(n_pixel, dtype=bool)
    good_pixels_mask[bad_pixels] = False
    good_pixels = pixels[good_pixels_mask]

    rate[bad_pixels] = avg_matrix[bad_pixels, :].dot(rate[good_pixels])
    if plot is None:
        return rate
    fig1, ax = plt.subplots(1, 1)
    display = CameraDisplay(DigiCam.geometry, ax=ax, norm=norm,
                            title='NSB rate [GHz]')
    rate_ghz = rate.to(u.GHz).value
    display.image = rate_ghz
    if plot_nsb_range is None:
        plot_nsb_range = (np.min(rate_ghz), np.max(rate_ghz))
    display.set_limits_minmax(*plot_nsb_range)
    display.add_colorbar(ax=ax)
    display.highlight_pixels(bad_pixels, color='r', linewidth=2)
    plt.tight_layout()
    output_path = os.path.dirname(plot)
    if plot == "show" or \
            (output_path != "" and not os.path.isdir(output_path)):
        if not plot == "show":
            print('WARNING: Path ' + output_path + ' for output trigger ' +
                  'uniformity does not exist, displaying the plot instead.\n')
        plt.show()
    else:
        plt.savefig(plot)
        print(plot, 'created')
    plt.close(fig1)
    return rate


def entry():
    args = docopt(__doc__)
    baseline_histo_file = args['<INPUT>']
    dark_histo_file = convert_text(args['--dark_hist'])
    param_file = convert_text(args['--parameters'])
    template_filename = convert_text(args['--template'])
    plot = convert_text(args['--plot'])
    bias_resistance = convert_float(args['--bias_resistance']) * u.Ohm
    cell_capacitance = convert_float(args['--cell_capacitance']) * u.Farad
    nsb_rate(
        baseline_histo_file, dark_histo_file, param_file, template_filename,
        plot=plot, bias_resistance=bias_resistance,
        cell_capacitance=cell_capacitance
    )


if __name__ == '__main__':
    entry()