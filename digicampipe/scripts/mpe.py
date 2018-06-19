#!/usr/bin/env python
'''
Do the Multiple Photoelectron anaylsis

Usage:
  mpe.py [options] [OUTPUT] [INPUT ...]

Options:
  -h --help                   Show this screen.
  --max_events=N              Maximum number of events to analyse
  -o OUTPUT --output=OUTPUT.  Folder where to store the results.
  -i INPUT --input=INPUT.     Input files.
  -c --compute                Compute the data.
  -f --fit                    Fit.
  -d --display                Display.
  -v --debug                  Enter the debug mode.
  -p --pixel=<PIXEL>          Give a list of pixel IDs.
  --ac_levels=<DAC>           LED AC DAC level
  --shift=N                   number of bins to shift before integrating
                              [default: 0].
  --integral_width=N          number of bins to integrate over
                              [default: 7].
  --save_figures              Save the plots to the OUTPUT folder
  --n_samples=N               Number of samples per waveform
  --bin_width=N               Bin width (in LSB) of the histogram
                              [default: 1]
'''
import os
from docopt import docopt
from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt

from digicampipe.io.event_stream import calibration_event_stream
from histogram.histogram import Histogram1D
from digicampipe.calib.camera.baseline import fill_digicam_baseline, \
    subtract_baseline
from digicampipe.calib.camera.peak import fill_pulse_indices
from digicampipe.calib.camera.charge import compute_charge, compute_amplitude
from digicampipe.utils.docopt import convert_max_events_args, \
    convert_pixel_args, convert_dac_level


def plot_event(events, pixel_id):

    for event in events:

        event.data.plot(pixel_id=pixel_id)
        plt.show()

        yield event


def compute(files, pixel_id, max_events, pulse_indices, integral_width,
            shift, bin_width, output_path,
            charge_histo_filename='charge_histo.pk',
            amplitude_histo_filename='amplitude_histo.pk',
            save=True):

    amplitude_histo_path = os.path.join(output_path, amplitude_histo_filename)
    charge_histo_path = os.path.join(output_path, charge_histo_filename)

    if os.path.exists(charge_histo_path) and save:

        raise IOError('File {} already exists'.format(charge_histo_path))

    if os.path.exists(amplitude_histo_path) and save:

        raise IOError('File {} already exists'.format(amplitude_histo_path))

    n_pixels = len(pixel_id)

    events = calibration_event_stream(files, pixel_id=pixel_id,
                                      max_events=max_events)
    # events = compute_baseline_with_min(events)
    events = fill_digicam_baseline(events)
    events = subtract_baseline(events)
    # events = find_pulse_with_max(events)
    events = fill_pulse_indices(events, pulse_indices)
    events = compute_charge(events, integral_width, shift)
    events = compute_amplitude(events)

    charge_histo = Histogram1D(
        data_shape=(n_pixels,),
        bin_edges=np.arange(-40 * integral_width,
                            4096 * integral_width,
                            bin_width),
        axis_name='reconstructed charge '
                  '[LSB $\cdot$ ns]'
    )

    amplitude_histo = Histogram1D(
        data_shape=(n_pixels,),
        bin_edges=np.arange(-40, 4096, 1),
        axis_name='reconstructed amplitude '
                  '[LSB]'
    )

    for event in events:
        charge_histo.fill(event.data.reconstructed_charge)
        amplitude_histo.fill(event.data.reconstructed_amplitude)

    if save:

        charge_histo.save(charge_histo_path)
        amplitude_histo.save(amplitude_histo_path)

    return amplitude_histo, charge_histo


def entry():

    args = docopt(__doc__)
    files = args['INPUT']
    debug = args['--debug']

    max_events = convert_max_events_args(args['--max_events'])
    output_path = args['OUTPUT']

    if not os.path.exists(output_path):

        raise IOError('Path for output does not exists \n')

    pixel_id = convert_pixel_args(args['--pixel'])
    integral_width = int(args['--integral_width'])
    shift = int(args['--shift'])
    bin_width = int(args['--bin_width'])
    n_samples = int(args['--n_samples'])  # TODO access this in a better way !
    ac_levels = convert_dac_level(args['--ac_levels'])
    timing_histo_filename = 'timing_histo.pk'
    timing_histo_filename = os.path.join(output_path, timing_histo_filename)

    timing_histo = Histogram1D.load(timing_histo_filename)

    n_pixels = len(pixel_id)
    n_ac_levels = len(ac_levels)

    if n_ac_levels != len(files):

        raise ValueError('n_ac levels = {} != '
                         'n_files = {}'.format(n_ac_levels, len(files)))

    if args['--compute']:

        amplitude = np.zeros((n_pixels, n_ac_levels))
        charge = np.zeros((n_pixels, n_ac_levels))
        time = np.zeros((n_pixels, n_ac_levels))

        for i, (file, ac_level) in tqdm(enumerate(zip(files, ac_levels)),
                                        total=n_ac_levels, desc='DAC level',
                                        leave=False):

            charge_histo_filename = 'charge_histo_ac_level_{}.pk' \
                                    ''.format(ac_level)
            amplitude_histo_filename = 'amplitude_histo_ac_level_{}.pk' \
                                       ''.format(ac_level)

            time[:, i] = timing_histo.mode()
            pulse_indices = time[:, i] // 4

            amplitude_histo, charge_histo = compute(
                    file,
                    pixel_id, max_events, pulse_indices, integral_width,
                    shift, bin_width, output_path,
                    charge_histo_filename=charge_histo_filename,
                    amplitude_histo_filename=amplitude_histo_filename,
                    save=True)

            amplitude[:, i] = amplitude_histo.mean()
            charge[:, i] = charge_histo.mean()

        plt.figure()
        plt.plot(amplitude[0], charge[0])
        plt.show()

        np.savez(os.path.join(output_path, 'mpe_results'),
                 amplitude=amplitude, charge=charge, time=time,
                 pixel_id=pixel_id, ac_levels=ac_levels)

    if args['--fit']:

        pass

    if args['--save_figures']:

        pass

    if args['--display']:

        amplitude_histo_path = os.path.join(output_path, 'amplitude_histo_ac_level_200.pk')
        charge_histo_path = os.path.join(output_path, 'charge_histo_ac_level_200.pk')

        charge_histo = Histogram1D.load(charge_histo_path)
        charge_histo.draw(index=(0,), log=False, legend=False)

        amplitude_histo = Histogram1D.load(amplitude_histo_path)
        amplitude_histo.draw(index=(0,), log=False, legend=False)
        plt.show()

        pass

    return


if __name__ == '__main__':

    entry()
