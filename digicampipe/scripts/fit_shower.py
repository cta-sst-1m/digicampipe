#!/usr/bin/env python
"""
Do Full Multiple Photoelectron anaylsis

Usage:
  digicam-fit-shower [options] [--] <INPUT>...

Options:
  -h --help                   Show this screen.
  --max_events=N              Maximum number of events to analyze
  -o OUTPUT --output=OUTPUT  Folder where to store the results.
  -v --debug                  Enter the debug mode.
  --ncall=N                   Number of calls for the fit [default: 10000]
"""

from digicampipe.utils.fitter import ShowerFitter, MPEShowerFitter
from ctapipe.image import hillas_parameters
from ctapipe.image.timing_parameters import timing_parameters
import numpy as np
import matplotlib.pyplot as plt
from digicampipe.io.event_stream import event_stream, calibration_event_stream
from digicampipe.visualization import EventViewer
from digicampipe.visualization.plot import plot_array_camera
from eventio import SimTelFile
from digicampipe.instrument.camera import DigiCam
from copy import copy
from digicampipe.utils.pulse_template import NormalizedPulseTemplate
from docopt import docopt
import pickle
from ctapipe.image.hillas import HillasParameterizationError
from digicampipe.image.hillas import compute_alpha
from pkg_resources import resource_filename
import os


def dict_to_dictnumpy(dictionary):

    for key, val in dictionary.items():

        dictionary[key] = np.array(val)

    return dictionary


def entry():
    args = docopt(__doc__)
    files = args['<INPUT>']
    debug = bool(args['--debug'])
    max_events = int(args['--max_events']) if args['--max_events'] is not None else None
    output_file = args['--output']

    # files = ['/sst1m/MC/simtel_krakow/gamma_100.0_300.0TeV_09.simtel.gz']
    geom = DigiCam.geometry

    hillas_names = ['charge', 't_cm', 'x_cm', 'y_cm',
                    'width', 'length', 'psi', 'v', 'id']
    true_values = {key: [] for key in hillas_names}
    hillas_values = [{key: [] for key in hillas_names},
                     {key: [] for key in hillas_names}]
    fit_values = [{key: [] for key in hillas_names},
                  {key: [] for key in hillas_names}]
    fit_values[0]['lh'] = []
    fit_values[1]['lh'] = []
    n_pixels = 1296
    n_call = 1000
    gain = 5.7 * np.ones(1296) * 0.791 # gain and gain drop
    sigma_e = np.ones(n_pixels) * 5
    sigma_s = np.ones(n_pixels) * 0.1045
    crosstalk = np.ones(n_pixels) * 0.08
    baseline = np.zeros(n_pixels)
    n_events = 0
    n_trigger = 0
    TEMPLATE_FILENAME = resource_filename(
        'digicampipe',
        os.path.join(
            'tests',
            'resources',
            'pulse_SST-1M_dark.txt'
        )
    )

    template = NormalizedPulseTemplate.load(TEMPLATE_FILENAME)
    # template.plot()
    # plt.show()
    n_past_events = 30
    rolling_mean = np.zeros((n_pixels, n_past_events)) * np.nan
    rolling_std = np.zeros((n_pixels, n_past_events)) * np.nan

    for event in calibration_event_stream(files, max_events=max_events):

        n_events += 1
        waveform = event.data.adc_samples
        digicam_baseline = event.data.digicam_baseline
        waveform = waveform - digicam_baseline[:, None]
        event_id = event.event_id

        rolling_mean = np.roll(rolling_mean, 1, axis=-1)
        rolling_mean[:, -1] = digicam_baseline
        baseline_mean = np.nanmean(rolling_mean, axis=-1)
        baseline_mean = baseline_mean[:, None] * np.ones(waveform.shape)

        rolling_std = np.roll(rolling_std, 1, axis=-1)
        rolling_std[:, -1] = np.sum(waveform**2, axis=-1)
        std = np.sum(rolling_std, axis=-1) / (waveform.shape[-1] * rolling_std.shape[-1] - 1)
        baseline_std = np.sqrt(std)
        baseline_std = baseline_std[:, None] * np.ones(waveform.shape)



        # method = 'Powell'
        method = 'L-BFGS-B'
        # method = 'TNC'
        # method = 'SLSQP'
        # method = 'trust-constr'
        # options = {'maxiter': n_call, 'maxfun': n_call}
        options = None #{'maxiter': n_call, 'maxfun': n_call}

        try:

            fitter_1 = ShowerFitter(data=waveform, gain=gain,
                                    baseline=baseline,
                                    crosstalk=crosstalk,
                                    template=template,
                                    error=baseline_std)

            fitter_2 = MPEShowerFitter(data=waveform,
                                       baseline=baseline,
                                       gain=gain,
                                       crosstalk=crosstalk,
                                       sigma_e=baseline_std,
                                       sigma_s=sigma_s,
                                       n_peaks=100,
                                       error=baseline_std,
                                       template=template)


            fitter_1.fit(method=method, verbose=debug, options=options)
            fitter_2.fit(method=method, verbose=debug, options=options)
            # fitters = [fitter_1, fitter_2]
            n_trigger += 1
            fitters = [fitter_2]
        except HillasParameterizationError as e:
            print('Could not fit event id {}'.format(event_id))
            print(e)
            continue

        for i, fitter in enumerate(fitters):

            for key, val in fitter.end_parameters.items():

                fit_values[i][key].append(val)
                hillas_values[i][key].append(fitter.start_parameters[key])
            fit_values[i]['lh'].append(fitter.likelihood(**fitter.end_parameters))
            fit_values[i]['id'].append(event_id)
            hillas_values[i]['id'].append(event_id)

            if debug:

                cam_display, _ = plot_array_camera(data=waveform.max(axis=-1))
                cam_display.add_ellipse(centroid=(fitter.start_parameters['x_cm'],
                                                  fitter.start_parameters['y_cm']),
                                        width=3 * fitter.start_parameters['width'],
                                        length=3 * fitter.start_parameters['length'],
                                        angle=fitter.start_parameters['psi'],
                                        linewidth=7, color='r')

                fitter.plot()
                fitter.plot_times()
                fitter.plot_waveforms()
                # fitter.template.plot()
                # fitter.plot_likelihood('x_cm', 'y_cm', size=(100, 100))
                # fitter.plot_likelihood('width', 'length', size=(100, 100))
                fitter.plot_1dlikelihood('charge')
                fitter.plot_1dlikelihood('x_cm')
                fitter.plot_1dlikelihood('y_cm')
                fitter.plot_1dlikelihood('psi')
                fitter.plot_1dlikelihood('width')
                fitter.plot_1dlikelihood('length')
                print(fitter)
                plt.show()
    data = {'fitted': fit_values, 'standard': hillas_values,
            'n_trigger': n_trigger, 'n_events': n_events}

    print(data['fitted'])
    print(data['n_trigger'], data['n_events'])

    with open(output_file, 'wb') as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
    # np.save('standard.npy', dict_to_dictnumpy(standard_values))
    # np.save('fit.npy', dict_to_dictnumpy(fit_values))

    # with open(output_file, 'rb') as f:
        # print(pickle.load(f))

def compute_true_parameters(event, geometry):
    true_pe = event['photoelectrons'][0]['photoelectrons']
    photoelectrons_times = event['photoelectrons'][0]['time']
    minimum_time = np.array([np.nanmin(t) for t in photoelectrons_times]).min()

    true_times = np.array([np.mean(t) for t in photoelectrons_times])
    true_hillas = hillas_parameters(geometry, true_pe)
    true_timing = timing_parameters(geometry, true_pe, true_times,
                                    true_hillas)
    return true_times, true_hillas, true_pe, true_timing


# data_stream = event_stream(files)

# viewer = EventViewer(data_stream)
# viewer.draw()

if __name__ == '__main__':
    entry()
