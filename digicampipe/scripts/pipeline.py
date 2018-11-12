#!/usr/bin/env python
"""
Run the standard pipeline up to Hillas parameters

Usage:
  digicam-pipeline [options] [--] <INPUT>...

Options:
  -h --help                   Show this screen.
  --max_events=N              Maximum number of events to analyze
  -o FILE --output=FILE       file where to store the results.
                              [Default: ./hillas.fits]
  --dark=FILE                 File containing the Histogram of
                              the dark analysis
  -v --debug                  Enter the debug mode.
  -c --compute
  -d --display
  -p --bad_pixels=LIST        Give a list of bad pixel IDs.
                              If "none", the bad pixels will be deduced from
                              the parameter file specified with --parameters.
                              [default: none]
  --shift=N                   number of bins to shift before integrating
                              [default: 0].
  --saturation_threshold=N    Threshold in LSB at which the pulse amplitude is
                              considered as saturated.
                              [default: 3000]
  --threshold_pulse=N         A threshold to which the integration of the pulse
                              is defined for saturated pulses.
                              [default: 0.1]
  --integral_width=N          number of bins to integrate over
                              [default: 7].
  --picture_threshold=N       Tailcut primary cleaning threshold
                              [Default: 30]
  --boundary_threshold=N      Tailcut secondary cleaning threshold
                              [Default: 15]
  --parameters=FILE           Calibration parameters file path
  --template=FILE             Pulse template file path
  --disable_bar               If used, the progress bar is not show while
                              reading files.
"""
import os
import astropy.units as u
import numpy as np
import pandas as pd
import yaml
from astropy.table import Table
from ctapipe.core import Field
from ctapipe.io.containers import HillasParametersContainer
from ctapipe.io.serializer import Serializer
from docopt import docopt
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from histogram.histogram import Histogram1D

from digicampipe.scripts.bad_pixels import get_bad_pixels
from digicampipe.calib import baseline, peak, charge, cleaning, image, tagging
from digicampipe.calib import filters
from digicampipe.instrument.camera import DigiCam
from digicampipe.io.event_stream import calibration_event_stream
from digicampipe.utils.docopt import convert_int, convert_list_int
from digicampipe.utils.pulse_template import NormalizedPulseTemplate
from digicampipe.visualization.plot import plot_array_camera
from digicampipe.image.hillas import compute_alpha, compute_miss, \
    correct_alpha_3


class PipelineOutputContainer(HillasParametersContainer):
    local_time = Field(int, 'event time')
    event_id = Field(int, 'event identification number')
    event_type = Field(int, 'event type')

    alpha = Field(float, 'Alpha parameter of the shower')
    miss = Field(float, 'Miss parameter of the shower')
    border = Field(bool, 'Is the event touching the camera borders')
    burst = Field(bool, 'Is the event during a burst')
    saturated = Field(bool, 'Is any pixel signal saturated')


def main(files, max_events, dark_filename, shift, integral_width,
         debug, hillas_filename, parameters_filename, compute, display,
         picture_threshold, boundary_threshold, template_filename,
         saturation_threshold, threshold_pulse,
         bad_pixels=None, disable_bar=False,):
    if compute:
        with open(parameters_filename) as file:
            calibration_parameters = yaml.load(file)
        if bad_pixels is None:
            bad_pixels = get_bad_pixels(parameters_filename, plot=None)

        pulse_template = NormalizedPulseTemplate.load(template_filename)

        pulse_area = pulse_template.integral() * u.ns
        ratio = pulse_template.compute_charge_amplitude_ratio(
            integral_width=integral_width, dt_sampling=4)  # ~ 0.24

        gain = np.array(calibration_parameters['gain'])  # ~ 20 LSB / p.e.
        gain_amplitude = gain * ratio

        crosstalk = np.array(calibration_parameters['mu_xt'])
        bias_resistance = 10 * 1E3 * u.Ohm  # 10 kOhm
        cell_capacitance = 50 * 1E-15 * u.Farad  # 50 fF
        geom = DigiCam.geometry

        dark_histo = Histogram1D.load(dark_filename)
        dark_baseline = dark_histo.mean()

        events = calibration_event_stream(files, max_events=max_events,
                                          disable_bar=disable_bar)
        events = baseline.fill_dark_baseline(events, dark_baseline)
        events = baseline.fill_digicam_baseline(events)
        events = tagging.tag_burst_from_moving_average_baseline(events)
        events = baseline.compute_baseline_shift(events)
        events = baseline.subtract_baseline(events)
        # events = baseline.compute_baseline_std(events, n_events=100)
        events = filters.filter_clocked_trigger(events)
        events = baseline.compute_nsb_rate(events, gain_amplitude,
                                           pulse_area, crosstalk,
                                           bias_resistance, cell_capacitance)
        events = baseline.compute_gain_drop(events, bias_resistance,
                                            cell_capacitance)
        events = peak.find_pulse_with_max(events)
        events = charge.compute_dynamic_charge(events,
                                               integral_width=integral_width,
                                               saturation_threshold=saturation_threshold,
                                               threshold_pulse=threshold_pulse,
                                               debug=debug,
                                               pulse_tail=False,)
        # events = charge.compute_charge(events, integral_width, shift)
        events = charge.interpolate_bad_pixels(events, geom, bad_pixels)
        events = charge.compute_photo_electron(events, gains=gain)
        # events = cleaning.compute_cleaning_1(events, snr=3)

        events = cleaning.compute_tailcuts_clean(
            events, geom=geom, overwrite=True,
            picture_thresh=picture_threshold,
            boundary_thresh=boundary_threshold, keep_isolated_pixels=False
        )
        events = cleaning.compute_boarder_cleaning(events, geom,
                                                   boundary_threshold)
        events = cleaning.compute_dilate(events, geom)

        events = image.compute_hillas_parameters(events, geom)

        # events = image.show(events, geom)

        output_file = Serializer(hillas_filename, mode='w', format='fits')

        data_to_store = PipelineOutputContainer()

        for event in events:

            if debug:
                print(event.hillas)
                print(event.data.nsb_rate)
                print(event.data.gain_drop)
                print(event.data.baseline_shift)
                print(event.data.border)
                plot_array_camera(np.max(event.data.adc_samples, axis=-1))
                plot_array_camera(np.nanmax(
                    event.data.reconstructed_charge, axis=-1))
                plot_array_camera(event.data.cleaning_mask.astype(float))
                plot_array_camera(event.data.reconstructed_number_of_pe)
                plt.show()

            data_to_store.alpha = compute_alpha(event.hillas)
            data_to_store.miss = compute_miss(event.hillas,
                                              data_to_store.alpha)
            data_to_store.local_time = event.data.local_time
            data_to_store.event_type = event.event_type
            data_to_store.event_id = event.event_id
            data_to_store.border = event.data.border
            data_to_store.burst = event.data.burst
            data_to_store.saturated = event.data.saturated

            for key, val in event.hillas.items():
                data_to_store[key] = val

            output_file.add_container(data_to_store)
        output_file.close()

    if display:

        data = Table.read(hillas_filename, format='fits')
        data = data.to_pandas()

        data['local_time'] = pd.to_datetime(data['local_time'])
        data = data.set_index('local_time')
        data = data.dropna()

        n_event = len(data['event_id'])

        is_cutted = np.logical_or(
            data['length'] / data['width'] >= 10.,
            data['length'] / data['width'] <= 2.
        )
        print('tagged', np.sum(data['burst']), '/', n_event,
              'events as of bad quality')
        print('tagged', np.sum(is_cutted), '/', n_event,
              'events cut by l/w')
        is_cutted = np.logical_or(
            is_cutted,
            data['length'] < 25
        )
        print('tagged', np.sum(is_cutted), '/', n_event,
              'events cut by 2 < l/w < 10 and l > 25 mm')
        is_cutted = np.logical_or(
            is_cutted,
            data['width'] < 15
        )
        print('tagged', np.sum(is_cutted), '/', n_event,
              'events cut by 2 < l/w < 10 and l > 25 mm and w > 15 mm')

        plt.figure(figsize=(9, 9))
        subplot = 0
        for key, val in data.items():
            if key in ['border', 'intensity', 'kurtosis', 'event_id',
                       'event_type', 'miss', 'burst']:
                continue
            subplot += 1
            print(subplot, '/', 9, 'plotting', key)
            plt.subplot(3, 3, subplot)
            val_split = [
                val[(~data['burst']) & (~is_cutted)],
                val[(~data['burst']) & is_cutted]
            ]
            plt.hist(val_split, bins='auto', stacked=True)
            plt.xlabel(key)
            if subplot == 1:
                plt.legend(['2 < l/w < 10', 'l/w cut'])
        plt.tight_layout()
        plt.savefig('hillas.png')
        plt.close()

        # 2d histogram of shower centers
        fig = plt.figure(figsize=(8, 8))
        plt.subplot(2, 2, 1)
        plt.hist2d(data['x'], data['y'], bins=100, norm=LogNorm())
        plt.ylabel('shower center Y [mm]')
        plt.xlabel('shower center X [mm]')
        plt.title('all events')
        cb = plt.colorbar()
        cb.set_label('Number of events')
        plt.axis('equal')
        plt.subplot(2, 2, 2)
        data_ok = data[(~data['burst'])]
        plt.hist2d(data_ok['x'], data_ok['y'], bins=100, norm=LogNorm())
        plt.ylabel('shower center Y [mm]')
        plt.xlabel('shower center X [mm]')
        plt.title('not burst')
        cb = plt.colorbar()
        cb.set_label('Number of events')
        plt.axis('equal')
        plt.subplot(2, 2, 3)
        data_ok = data[(~is_cutted)]
        plt.hist2d(data_ok['x'], data_ok['y'], bins=100, norm=LogNorm())
        plt.ylabel('shower center Y [mm]')
        plt.xlabel('shower center X [mm]')
        plt.title('2 < l/w < 10')
        cb = plt.colorbar()
        cb.set_label('Number of events')
        plt.axis('equal')
        plt.subplot(2, 2, 4)
        data_ok = data[(~data['burst']) & (~is_cutted)]
        plt.hist2d(data_ok['x'], data_ok['y'], bins=100, norm=LogNorm())
        plt.ylabel('shower center Y [mm]')
        plt.xlabel('shower center X [mm]')
        plt.title('pass all')
        cb = plt.colorbar()
        cb.set_label('Number of events')
        plt.axis('equal')
        plt.tight_layout()
        plt.savefig('shower_center_map.png')
        plt.close(fig)

        # correlation plot
        fig = plt.figure(figsize=(12, 9))
        for title, data_pl in zip(['all', 'pass cuts'], [data, data_ok]):
            fig.clear()
            subplot = 0
            for i, (label_x, x) in enumerate(zip(
                ['shower center X [mm]', 'shower center Y [mm]'],
                [data_pl['x'], data_pl['y']]
            )):
                for j, (label_y, y, ymin, ymax) in enumerate(zip(
                    [
                        'shower length [mm]',
                        'shower width [mm]',
                        'length/width',
                        'r - l/2 [mm]'
                    ],
                    [
                        data_pl['length'],
                        data_pl['width'],
                        data_pl['length']/data_pl['width'],
                        data_pl['r'] - data_pl['length']/2
                    ],
                    [0, 0, 0, -100],
                    [200, 100, 10, 500]
                )):
                    subplot += 1
                    plt.subplot(2, 4, subplot)
                    plt.hist2d(x, y, bins=(100, np.linspace(ymin, ymax, 100)),
                               norm=LogNorm())
                    plt.ylim(ymin, ymax)
                    plt.xlabel(label_x)
                    plt.ylabel(label_y)
                    plt.title(title)
                    cb = plt.colorbar()
                    cb.set_label('Number of events')
            plt.tight_layout()
            plt.savefig('correlation_{}.png'.format(title.replace(' ', '_')))
        plt.close(fig)

        # 2D scan of spike in alpha
        bin_size = 4  # binning in degrees
        num_steps = 80  # number of binning in the FoV
        x_fov_start = -500  # limits of the FoV
        y_fov_start = -500  # limits of the FoV
        x_fov_end = 500  # limits of the FoV
        y_fov_end = 500  # limits of the FoV
        mask = (~data['border']) & (~data['burst']) & (
                data['length']/data['width'] > 1.5) & (
                data['length']/data['width'] < 10) & (
                data['length'] > 25) & (data['width'] > 15)
        data_cor = dict()
        for key, val in data.items():
            data_cor[key] = val[mask]
        x_fov = np.linspace(x_fov_start, x_fov_end, num_steps)
        y_fov = np.linspace(y_fov_start, y_fov_end, num_steps)
        dx = x_fov[1] - x_fov[0]
        dy = y_fov[1] - y_fov[0]
        x_fov_bins = np.linspace(x_fov_start - dx / 2, x_fov_end + dx / 2,
                                 num_steps + 1)
        y_fov_bins = np.linspace(y_fov_start - dy / 2, y_fov_end + dy / 2,
                                 num_steps + 1)
        N = np.zeros([num_steps, num_steps], dtype=int)
        i = 0
        print('2D scan calculation:')
        for xi, x in enumerate(x_fov):
            print(round(i / len(x_fov) * 100, 2), '/', 100)  # progress
            for yi, y in enumerate(y_fov):
                data_cor2 = correct_alpha_3(data_cor, source_x=x, source_y=y)
                mask2 = data_cor2['alpha'] < bin_size
                alpha_filtered = data_cor2['alpha'][mask2]
                N[yi, xi] = alpha_filtered.shape[0]
            i += 1
        fig = plt.figure(figsize=(8, 6))
        ax1 = fig.add_subplot(111)
        pcm = ax1.pcolormesh(x_fov_bins, y_fov_bins, N,
                             rasterized=True, cmap='nipy_spectral')
        plt.ylabel('FOV Y [mm]')
        plt.xlabel('FOV X [mm]')
        cbar = fig.colorbar(pcm)
        cbar.set_label('N of events')
        plt.savefig('2d_alpha_scan.png')


def entry():
    args = docopt(__doc__)
    files = args['<INPUT>']
    max_events = convert_int(args['--max_events'])
    dark_filename = args['--dark']
    output = args['--output']
    compute = args['--compute']
    display = args['--display']
    output_path = os.path.dirname(output)
    if output_path != "" and not os.path.exists(output_path):
        raise IOError('Path ' + output_path +
                      'for output hillas does not exists \n')
    bad_pixels = convert_list_int(args['--bad_pixels'])
    integral_width = int(args['--integral_width'])
    picture_threshold = float(args['--picture_threshold'])
    boundary_threshold = float(args['--boundary_threshold'])
    shift = int(args['--shift'])
    debug = args['--debug']
    parameters_filename = args['--parameters']
    template_filename = args['--template']
    disable_bar = args['--disable_bar']
    saturation_threshold = float(args['--saturation_threshold'])
    threshold_pulse = float(args['--threshold_pulse'])
    main(files=files,
         max_events=max_events,
         dark_filename=dark_filename,
         shift=shift,
         integral_width=integral_width,
         debug=debug,
         parameters_filename=parameters_filename,
         hillas_filename=output,
         compute=compute,
         display=display,
         picture_threshold=picture_threshold,
         boundary_threshold=boundary_threshold,
         template_filename=template_filename,
         bad_pixels=bad_pixels,
         disable_bar=disable_bar,
         threshold_pulse=threshold_pulse,
         saturation_threshold=saturation_threshold,
         )


if __name__ == '__main__':
    entry()
