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
  -p --pixel=<PIXEL>          Give a list of pixel IDs.
  --shift=N                   number of bins to shift before integrating
                              [default: 0].
  --integral_width=N          number of bins to integrate over
                              [default: 7].
  --picture_threshold=N       Tailcut primary cleaning threshold
                              [Default: 20]
  --boundary_threshold=N      Tailcut secondary cleaning threshold
                              [Default: 15]
  --parameters=FILE           Calibration parameters file path
  --template=FILE             Pulse template file path
  --burst=FILE                File with burst events. If none, no bursts are
                              considered.
                              [Default: none]
"""
import matplotlib
matplotlib.use("Agg")
from digicampipe.io.event_stream import calibration_event_stream
from ctapipe.io.serializer import Serializer
from ctapipe.io.containers import HillasParametersContainer
from ctapipe.core import Field
from digicampipe.utils.docopt import convert_max_events_args, \
    convert_pixel_args
from digicampipe.calib.camera import baseline, peak, charge, cleaning, image, \
    filter
from digicampipe.utils.pulse_template import NormalizedPulseTemplate
from digicampipe.utils import DigiCam
import os
import yaml
from docopt import docopt
from histogram.histogram import Histogram1D
from astropy.table import Table
import astropy.units as u
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import pandas as pd
from digicampipe.visualization.plot import plot_array_camera
from digicampipe.utils.hillas import compute_alpha, compute_miss, \
    correct_alpha_3


class PipelineOutputContainer(HillasParametersContainer):

    local_time = Field(int, 'event time')
    event_id = Field(int, 'event identification number')
    event_type = Field(int, 'event type')

    alpha = Field(float, 'Alpha parameter of the shower')
    miss = Field(float, 'Miss parameter of the shower')
    border = Field(bool, 'Is the event touching the camera borders')


def main(files, max_events, dark_filename, pixel_ids, shift, integral_width,
         debug, hillas_filename, parameters_filename, compute, display,
         picture_threshold, boundary_threshold, template_filename,
         burst_filename):

    if compute:

        with open(parameters_filename) as file:

            calibration_parameters = yaml.load(file)

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

        events = calibration_event_stream(files, pixel_id=pixel_ids,
                                          max_events=max_events)
        events = baseline.fill_dark_baseline(events, dark_baseline)
        events = baseline.fill_digicam_baseline(events)
        events = baseline.compute_baseline_shift(events)
        events = baseline.subtract_baseline(events)
        # events = baseline.compute_baseline_std(events, n_events=100)
        events = filter.filter_clocked_trigger(events)
        events = baseline.compute_nsb_rate(events, gain_amplitude,
                                           pulse_area, crosstalk,
                                           bias_resistance, cell_capacitance)
        events = baseline.compute_gain_drop(events, bias_resistance,
                                            cell_capacitance)
        events = peak.find_pulse_with_max(events)
        events = charge.compute_charge(events, integral_width, shift)
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
        is_burst = np.zeros(n_event, dtype=bool)
        is_cutted = np.zeros(n_event, dtype=bool)
        if burst_filename != "none":
            bursts = [[], []]
            with open(burst_filename, 'r') as fd:
                lines = fd.readlines()
                for line in lines:
                    line = line.strip()
                    if len(line) < 1 or line[0] == '#':
                        continue
                    splited_line = line.split(' ')
                    if len(splited_line) != 5:
                        print('WARNING, in burst file', burst_filename,
                              'line "', line, '" invalid.')
                        continue
                    burst, ts_start, ts_end, id_start, id_end = splited_line
                    bursts[0].append(ts_start)
                    bursts[1].append(ts_end)
            n_burst = len(bursts[0])
            bursts[0] = pd.to_datetime(bursts[0])
            bursts[1] = pd.to_datetime(bursts[1])
            ts = np.int64(data.index)
            for burst_idx in range(n_burst):
                burst_t_min = bursts[0][burst_idx].value
                burst_t_max = bursts[1][burst_idx].value

                in_interval = np.logical_and(
                    ts <= burst_t_max,
                    burst_t_min <= ts
                )
                is_burst = np.logical_or(
                    is_burst,
                    in_interval
                )
            is_cutted = np.logical_or(
                data['length'] / data['width'] >= 10.,
                data['length'] / data['width'] <= 2.
            )
            print('tagged', np.sum(is_burst), '/', n_event,
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
            if key == 'border':
                continue
            if key == 'intensity':
                continue
            if key == 'kurtosis':
                continue
            if key == 'event_id':
                continue
            if key == 'event_type':
                continue
            if key == 'miss':
                continue
            subplot += 1
            print(subplot, '/', 9, 'plotting', key)
            plt.subplot(3, 3, subplot)
            val_split = [
                val[(is_burst == False) & (is_cutted == False)],
                val[(is_burst == False) & (is_cutted == True)]
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
        data_ok = data[(is_burst == False)]
        plt.hist2d(data_ok['x'], data_ok['y'], bins=100, norm=LogNorm())
        plt.ylabel('shower center Y [mm]')
        plt.xlabel('shower center X [mm]')
        plt.title('not burst')
        cb = plt.colorbar()
        cb.set_label('Number of events')
        plt.axis('equal')
        plt.subplot(2, 2, 3)
        data_ok = data[(is_cutted == False)]
        plt.hist2d(data_ok['x'], data_ok['y'], bins=100, norm=LogNorm())
        plt.ylabel('shower center Y [mm]')
        plt.xlabel('shower center X [mm]')
        plt.title('2 < l/w < 10')
        cb = plt.colorbar()
        cb.set_label('Number of events')
        plt.axis('equal')
        plt.subplot(2, 2, 4)
        data_ok = data[(is_burst == False) & (is_cutted == False)]
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
        for title, data_pl in zip(['all','pass cuts'], [data, data_ok]):
            fig.clear()
            subplot = 0
            for i, (label_x, x) in enumerate(zip(
                ['shower center X [mm]', 'shower center Y [mm]'],
                [data_pl['x'], data_pl['y']]
            )):
                for j, (label_y, y, ymin, ymax) in enumerate(zip(
                    ['shower length [mm]', 'shower width [mm]', 'length/width', 'r - l/2 [mm]'],
                    [data_pl['length'], data_pl['width'], data_pl['length']/data_pl['width'], data_pl['r'] - data_pl['length']/2],
                    [0, 0, 0, -100],
                    [200, 100, 10, 500]
                )):
                    # print('creating', label_y, 'vs', label_x, 'plot for', title)
                    subplot+=1
                    plt.subplot(2, 4, subplot)
                    plt.hist2d(x, y, bins=(100, np.linspace(ymin, ymax, 100)), norm=LogNorm())
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
        data['burst'] = is_burst
        bin_size = 4  # binning in degrees
        num_steps = 160 #60  # number of binning in the FoV
        x_fov_start = -400  # limits of the FoV
        y_fov_start = -400  # limits of the FoV
        x_fov_end = 400  # limits of the FoV
        y_fov_end = 400  # limits of the FoV
        mask = (data['border'] == False) & (data['length'] > 25) & (
                data['length']/data['width'] > 2) & (
                data['length']/data['width'] < 10) & (
                data['burst'] == False) & (
                data['width'] > 15)
        data_cor = dict()
        for key, val in data.items():
            data_cor[key] = val[mask]
        x_fov = np.linspace(x_fov_start, x_fov_end, num_steps)
        y_fov = np.linspace(y_fov_start, y_fov_end, num_steps)
        dx = x_fov[1] - x_fov[0]
        dy = y_fov[1] - y_fov[0]
        x_fov_bins = np.linspace(x_fov_start - dx / 2, x_fov_end + dx / 2,
                                  num_steps + 1)
        y_fov_bins = np.linspace(y_fov_start - dy/ 2, y_fov_end + dy / 2,
                                  num_steps + 1)
        N = np.zeros([num_steps, num_steps], dtype=int)
        i = 0
        print('2D scan calculation:')
        for xi, x in enumerate(x_fov):
            print(round(i / len(x_fov) * 100, 2), '/', 100)  # progress
            for yi, y in enumerate(y_fov):
                data_cor2 = correct_alpha_3(data_cor, source_x=x, source_y=y)
                mask2 = (data_cor2['alpha'] < bin_size) & (
                        data_cor2['r'] - data_cor2['length'] / 2.0 > 0)
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
    max_events = convert_max_events_args(args['--max_events'])
    dark_filename = args['--dark']
    output = args['--output']
    compute = args['--compute']
    display = args['--display']
    output_path = os.path.dirname(output)

    if not os.path.exists(output_path):
        raise IOError('Path ' + output_path +
                      'for output hillas does not exists \n')
    pixel_ids = convert_pixel_args(args['--pixel'])
    integral_width = int(args['--integral_width'])
    picture_threshold = float(args['--picture_threshold'])
    boundary_threshold = float(args['--boundary_threshold'])
    shift = int(args['--shift'])
    debug = args['--debug']
    parameters_filename = args['--parameters']
    # args['--min_photon'] = int(args['--min_photon'])
    template_filename = args['--template']
    burst_filename = args['--burst']
    main(files=files,
         max_events=max_events,
         dark_filename=dark_filename,
         pixel_ids=pixel_ids,
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
         burst_filename=burst_filename
         )


if __name__ == '__main__':
    entry()
