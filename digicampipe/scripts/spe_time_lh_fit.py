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
  --mc                        Is MC file [default: True]
  --boundary_threshold=N      Boundary threshold [default: 10]
  --picture_threshold=N       Picture threshold [default: 10]
"""

from digicampipe.utils.fitter import SpaceTimeFitter, PoissonSpaceTimeFitter, HillasFitter, SPESpaceTimeFitter
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
from ctapipe.core.container import Container, Field
import astropy.units as u
from ctapipe.io.hdf5tableio import HDF5TableWriter
from digicampipe.io.containers import ImageParametersContainer


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
    mc = bool(args['--mc'])

    geometry = DigiCam.geometry

    print(debug, "HELLO")
    n_pixels = 1296
    n_call = 10000
    gain = 5.7 * np.ones(1296) * 0.791 # gain and gain drop (Jakub simulations)
    # gain = 5.43 * np.ones(1296) # * 0.791 # gain and gain drop (Prod4b)
    # gain = np.ones(1296) * 10 # gain and gain drop
    sigma_s = np.ones(n_pixels) * 0.104571984
    # sigma_s = np.ones(n_pixels) * 1
    crosstalk = np.ones(n_pixels) * 0.08
    dt = 4
    integral_width = 7
    baseline = np.zeros(n_pixels)
    n_events = 0
    n_trigger = 0

    TEMPLATE_FILENAME = resource_filename('digicampipe', os.path.join( 'tests', 'resources', 'pulse_SST-1M_dark.txt'))
    # TEMPLATE_FILENAME = resource_filename('digicampipe', os.path.join( 'tests', 'resources', 'pulse_SST-1M_pixel_0.dat'))
    template = NormalizedPulseTemplate.load(TEMPLATE_FILENAME)

    sigma_space = 3
    sigma_time = 4
    sigma_amplitude = 3
    picture_threshold = float(args['--picture_threshold'])
    boundary_threshold = float(args['--boundary_threshold'])
    time_before_shower = 20
    time_after_shower = 40
    n_peaks = 500

    # template.plot()
    # plt.show()
    baseline = np.zeros(n_pixels)
    baseline_std = np.zeros(n_pixels)
    container = ImageParametersContainer()
    init_containter = ImageParametersContainer()

    print("Computing baseline fluctuations")
    N = 0
    for event in calibration_event_stream(files):
        waveform = event.data.adc_samples
        digicam_baseline = event.data.digicam_baseline
        waveform = waveform - digicam_baseline[:, None]
        event_id = event.event_id

        baseline += np.sum(waveform, axis=-1)
        baseline_std += np.sum(waveform ** 2, axis=-1)

        N += 1
    N = N * waveform.shape[-1]
    baseline = baseline / N
    baseline_std = np.sqrt(baseline_std / N - baseline**2)
    baseline_std = np.median(baseline_std) * np.ones(waveform.shape)
    baseline = np.zeros(waveform.shape[0])
    print("Writing data to {}".format(output_file))
    with HDF5TableWriter(output_file, 'data') as f:

        for event in calibration_event_stream(files):

            n_events += 1
            waveform = event.data.adc_samples
            digicam_baseline = event.data.digicam_baseline
            waveform = waveform - digicam_baseline[:, None]
            event_id = event.event_id


            # plot_array_camera(waveform.max(axis=-1))
            # plt.show()
            # continue

            # method = 'Powell'
            method = 'L-BFGS-B'
            minuit = True
            # method = 'TNC'
            # method = 'SLSQP'
            # method = 'trust-constr'
            # options = {'maxiter': n_call, 'maxfun': n_call}
            options = None
            options = {'minuit': minuit, 'maxiter': n_call, 'maxfun': n_call}
            config = dict(data=waveform,
                          error=baseline_std,
                          gain=gain, baseline=baseline,
                          crosstalk=crosstalk, sigma_s=sigma_s,
                          geometry=geometry, dt=dt,
                          integral_width=integral_width,
                          template=template,
                          sigma_space=sigma_space,
                          sigma_time=sigma_time,
                          sigma_amplitude=sigma_amplitude,
                          picture_threshold=picture_threshold,
                          boundary_threshold=boundary_threshold,
                          time_before_shower=time_before_shower,
                          time_after_shower=time_after_shower)

            fitted = True
            try:

                fitter = SPESpaceTimeFitter(**config, n_peaks=n_peaks)
                fitter.fit(method=method, verbose=debug, **options)
                container = fitter.to_container()
                init_containter = fitter.to_container(start_parameters=True)

                n_trigger += 1

            except (HillasParameterizationError, ValueError) as e:
                fitted = False
                print('Could not fit event id {} and telescope id {}'.format(event_id, event.tel_id))
                print(e)

            container.event_id = event_id
            container.true_energy = event.mc.energy
            container.particle = event.mc.shower_primary_id
            container.tel_id = event.tel_id
            container.alt = event.alt
            container.az = event.az
            container.tel_alt = event.tel_alt
            container.tel_az = event.tel_az
            container.x_max = event.x_max
            container.core_x = event.core_x
            container.core_y = event.core_y
            container.h_first = event.h_first



            f.write('mc', container)
            f.write('hillas', container.hillas)
            f.write('timing', container.timing)
            f.write('init_hillas', init_containter.hillas)
            f.write('init_timing', init_containter.timing)

            if debug and (n_trigger >= 1) and fitted:

                print('Event ID ', event_id, ' Telescope ID ',  event.tel_id)

                try:

                    print(container)

                    fitter.plot()
                    # fitter.plot_times()
                    fitter.plot_waveforms()
                    fitter.plot_ellipse()
                    fitter.plot_times_camera()
                    fitter.plot_waveforms_3D()
                    fitter.plot_1dlikelihood('t_cm')
                    fitter.plot_1dlikelihood('v')
                    fitter.plot_1dlikelihood('charge')
                    fitter.plot_1dlikelihood('psi')
                    fitter.plot_1dlikelihood('x_cm')
                    fitter.plot_1dlikelihood('y_cm')
                    fitter.plot_1dlikelihood('width')
                    fitter.plot_1dlikelihood('length')

                    plt.show()
                except ValueError:
                    pass

            container.reset()
            init_containter.reset()

            if max_events is not None:
                if n_trigger >= max_events:

                    break


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
