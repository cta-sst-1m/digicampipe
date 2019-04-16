#!/usr/bin/env python
"""
Run the standard pipeline up to Hillas parameters

Usage:
  digicam-pipeline [options] [--] <INPUT>...

Options:
  -h --help                     Show this screen.
  <INPUT>                       List of zfits input files. Typically a single
                                night observing a single source.
  --aux_basepath=DIR            Base directory for the auxilary data.
                                If set to "search", it will try to determine it
                                from the path of the first input file. If set
                                to "none", no auxiliary data will be added.
                                [Default: search]
  --max_events=N                Maximum number of events to analyze
  -o FILE --output=FILE         file where to store the results.
                                [Default: ./hillas.fits]
  --dark=FILE                   File containing the Histogram of
                                the dark analysis
  -v --debug                    Enter the debug mode.
  -p --bad_pixels=LIST          Give a list of bad pixel IDs.
                                If "none", the bad pixels will be deduced from
                                the parameter file specified with --parameters.
                                [default: none]
  --saturation_threshold=N      Threshold in LSB at which the pulse amplitude
                                is considered as saturated.
                                [default: 3000.]
  --threshold_pulse=N           A threshold to which the integration of the
                                pulse is defined for saturated pulses.
                                [default: 0.1]
  --integral_width=INT          Number of bins to integrate over
                                [default: 7].
  --picture_threshold=N         Tailcut primary cleaning threshold
                                [Default: 30.]
  --boundary_threshold=N        Tailcut secondary cleaning threshold
                                [Default: 15.]
  --parameters=FILE             Calibration parameters file path
  --template=FILE               Pulse template file path
  --nevent_plot=INT             number of example events to plot
                                [Default: 12]
  --event_plot_filename=PATH    name of the image created when displaying the
                                number of event specified by --print_nevent .
                                If set to display, the plot is shown instead of
                                being saved. If set to "none" no event are
                                shown. [Default: none]
  --disable_bar                 If used, the progress bar is not show while
                                reading files.
  --wdw_number=INT              Window that was used for the measurement.
                                [default: 1].
  --apply_corr_factor           If used, correction factors corresponding
                                to the window non-uniformity are applied.
"""
import os
import sys
import astropy.units as u
import numpy as np
import yaml
from ctapipe.core import Field
from ctapipe.io.containers import HillasParametersContainer
from ctapipe.io.serializer import Serializer
from ctapipe.visualization import CameraDisplay
from ctapipe.image.cleaning import number_of_islands
from docopt import docopt
import matplotlib.pyplot as plt
from histogram.histogram import Histogram1D

from digicampipe.scripts.bad_pixels import get_bad_pixels
from digicampipe.calib import baseline, peak, charge, cleaning, image, tagging
from digicampipe.calib import filters
from digicampipe.instrument.camera import DigiCam
from digicampipe.io.event_stream import calibration_event_stream, \
    add_slow_data_calibration
from digicampipe.utils.docopt import convert_int, convert_list_int, \
    convert_text, convert_float
from digicampipe.utils.pulse_template import NormalizedPulseTemplate
from digicampipe.visualization.plot import plot_array_camera
from digicampipe.image.hillas import compute_alpha, compute_miss


class PipelineOutputContainer(HillasParametersContainer):
    # info on event
    local_time = Field(np.int64, 'Event time in nanoseconds since 1970')
    event_id = Field(int, 'Event identification number')
    event_type = Field(int, 'Event type')
    az = Field(float, 'Current azimuth position info from DriveSystem')
    el = Field(float, 'Current elevation position info from DriveSystem')
    # additional Hillas parameters
    alpha = Field(float, 'Alpha parameter of the shower')
    miss = Field(float, 'Miss parameter of the shower')
    # data quality values
    baseline = Field(float, 'Baseline average over the camera')
    nsb_rate = Field(float, 'NSB rate averaged over the camera')
    digicam_temperature = Field(float, 'average DigiCam temperature')
    pdp_temperature = Field(float, 'SiPM temperature averaged over the camera')
    target_ra = Field(float, 'Right ascension of the current target')
    target_dec = Field(float, 'Declination of the current target')
    number_of_island = Field(int, 'Number of islands after tail-cut cleaning')
    # data quality flags
    pointing_leds_on = Field(bool, 'Are the pointing LEDs on (continuous)')
    pointing_leds_blink = Field(bool, 'Are the pointing LEDs blinking')
    all_hv_on = Field(bool, 'Are all the HV on for the SiPMs')
    all_ghv_on = Field(bool, 'Are all the GHV on for the SiPMs')
    is_on_source = Field(bool, 'Is the telescope on source')
    is_tracking = Field(bool, 'Is the telescope tracking')
    shower = Field(bool, 'Is the event a shower according to 3d algorithm')
    border = Field(bool, 'Is the event touching the camera borders')
    burst = Field(bool, 'Is the event during a burst')
    saturated = Field(bool, 'Is any pixel signal saturated')


def plot_nevent(events, nevent, filename, bad_pixels=None, norm="lin"):
    displays = []
    fig, axes = plt.subplots(3, 4,  # sharex='all', sharey='all',
                             figsize=[18, 12])
    axes = axes.flatten()
    for index, event in enumerate(events):
        if index < nevent:
            axe = index % 12
            figure = int(np.floor(index / 12))
            if index < 12:
                displays.append(
                    CameraDisplay(DigiCam.geometry, ax=axes[index], norm=norm,
                                  title='')
                )
                displays[axe].cmap.set_bad('w')
                displays[axe].cmap.set_over('w')
                displays[axe].cmap.set_under('w')
                displays[axe].add_colorbar(ax=axes[axe])
                axes[axe].set_xlabel("")
                axes[axe].set_ylabel("")
                axes[axe].set_xlim([-400, 400])
                axes[axe].set_ylim([-400, 400])
                axes[axe].set_xticklabels([])
                axes[axe].set_yticklabels([])
            pe = event.data.reconstructed_number_of_pe
            n_pix = len(pe)
            mask = event.data.cleaning_mask
            pe_masked = pe
            pe_masked[~mask] = np.NaN
            displays[axe].set_limits_minmax(0, np.nanmax(pe_masked))
            displays[axe].image = pe_masked
            # highlight only bad pixels which pass the tail-cut cleaning
            highlighted_mask = np.zeros(n_pix, dtype=bool)
            highlighted_mask[bad_pixels] = mask[bad_pixels]
            highlighted = np.arange(n_pix)[highlighted_mask]
            displays[axe].highlight_pixels(highlighted, color='k', linewidth=2)
            displays[axe].overlay_moments(event.hillas, with_label=False,
                                          edgecolor='r', linewidth=2)
            if index % 12 == 11 or index == nevent - 1:
                if filename.lower == "show":
                    plt.show()
                else:
                    if figure == 0:
                        output = filename
                    else:
                        output = filename.replace('.png',
                                                  '_' + str(figure) + '.png')
                    plt.savefig(output)
                    print(output, 'created.')
        yield event
    plt.close(fig)


def main_pipeline(
        files, aux_basepath, max_events, dark_filename, integral_width,
        debug, hillas_filename, parameters_filename,
        picture_threshold, boundary_threshold, template_filename,
        saturation_threshold, threshold_pulse, nevent_plot=12,
        event_plot_filename=None, bad_pixels=None, disable_bar=False,
        wdw_number=1, apply_corr_factor=False,
):
    # get configuration
    with open(parameters_filename) as file:
        calibration_parameters = yaml.load(file)
    if bad_pixels is None:
        bad_pixels = get_bad_pixels(
            calib_file=parameters_filename,
            dark_histo=dark_filename,
            plot=None
        )
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

    # define pipeline
    events = calibration_event_stream(files, max_events=max_events,
                                      disable_bar=disable_bar)
    if aux_basepath is not None:
        events = add_slow_data_calibration(
            events, basepath=aux_basepath,
            aux_services=('DriveSystem', 'DigicamSlowControl', 'MasterSST1M',
                          'SafetyPLC', 'PDPSlowControl')
        )
    events = baseline.fill_dark_baseline(events, dark_baseline)
    events = baseline.fill_digicam_baseline(events)
    events = tagging.tag_burst_from_moving_average_baseline(events)
    events = baseline.compute_baseline_shift(events)
    events = baseline.subtract_baseline(events)
    events = filters.filter_clocked_trigger(events)
    events = baseline.compute_nsb_rate(events, gain_amplitude,
                                       pulse_area, crosstalk,
                                       bias_resistance, cell_capacitance)
    events = baseline.compute_gain_drop(events, bias_resistance,
                                        cell_capacitance)
    events = peak.find_pulse_with_max(events)
    events = charge.compute_dynamic_charge(
        events,
        integral_width=integral_width,
        saturation_threshold=saturation_threshold,
        threshold_pulse=threshold_pulse,
        debug=debug,
        pulse_tail=False,
    )
    events = charge.compute_photo_electron(events, gains=gain)
    events = charge.apply_wdw_transmittance_correction_factor(
        events, wdw_number, apply_corr_factor
    )
    events = charge.interpolate_bad_pixels(events, geom, bad_pixels)
    events = cleaning.compute_tailcuts_clean(
        events, geom=geom, overwrite=True,
        picture_thresh=picture_threshold,
        boundary_thresh=boundary_threshold, keep_isolated_pixels=False
    )
    events = cleaning.compute_boarder_cleaning(events, geom,
                                               boundary_threshold)
    events = cleaning.compute_dilate(events, geom)
    events = image.compute_hillas_parameters(events, geom)
    if event_plot_filename is not None:
        events = plot_nevent(events, nevent_plot, filename=event_plot_filename,
                             bad_pixels=bad_pixels, norm="lin")
    events = charge.compute_sample_photo_electron(events, gain_amplitude)
    events = cleaning.compute_3d_cleaning(
        events, geom, n_sample=50, threshold_sample_pe=20,
        threshold_time=2.1 * u.ns, threshold_size=0.005 * u.mm
    )
    # create pipeline output file
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
        # fill container
        data_to_store.local_time = event.data.local_time
        data_to_store.event_type = event.event_type
        data_to_store.event_id = event.event_id
        r = event.hillas.r
        phi = event.hillas.phi
        psi = event.hillas.psi
        alpha = compute_alpha(phi.value, psi.value) * u.rad
        data_to_store.alpha = alpha
        data_to_store.miss = compute_miss(r=r.value, alpha=alpha.value)
        data_to_store.miss = data_to_store.miss * r.unit
        data_to_store.baseline = np.mean(event.data.digicam_baseline)
        data_to_store.nsb_rate = np.mean(event.data.nsb_rate)
        data_to_store.shower = bool(event.data.shower)
        data_to_store.border = bool(event.data.border)
        data_to_store.burst = bool(event.data.burst)
        data_to_store.saturated = bool(event.data.saturated)
        num_islands, island_labels = number_of_islands(
            geom, event.data.cleaning_mask
        )
        data_to_store.number_of_island = num_islands
        if aux_basepath is not None:
            data_to_store.az = event.slow_data.DriveSystem.current_position_az
            data_to_store.el = event.slow_data.DriveSystem.current_position_el
            temp_crate1 = event.slow_data.DigicamSlowControl.Crate1_T
            temp_crate2 = event.slow_data.DigicamSlowControl.Crate2_T
            temp_crate3 = event.slow_data.DigicamSlowControl.Crate3_T
            temp_digicam = np.array(
                np.hstack([temp_crate1, temp_crate2, temp_crate3])
            )
            temp_digicam_mean = np.mean(
                temp_digicam[np.logical_and(temp_digicam > 0,
                                            temp_digicam < 60)]
            )
            data_to_store.digicam_temperature = temp_digicam_mean
            temp_sector1 = event.slow_data.PDPSlowControl.Sector1_T
            temp_sector2 = event.slow_data.PDPSlowControl.Sector2_T
            temp_sector3 = event.slow_data.PDPSlowControl.Sector3_T
            temp_pdp = np.array(
                np.hstack([temp_sector1, temp_sector2, temp_sector3])
            )
            temp_pdp_mean = np.mean(
                temp_pdp[np.logical_and(temp_pdp > 0, temp_pdp < 60)]
            )
            data_to_store.pdp_temperature = temp_pdp_mean
            target_radec = event.slow_data.MasterSST1M.target_radec
            data_to_store.target_ra = target_radec[0]
            data_to_store.target_dec = target_radec[1]
            status_leds = event.slow_data.SafetyPLC.SPLC_CAM_Status
            # bit 8 of status_LEDs is about on/off, bit 9 about blinking
            data_to_store.pointing_leds_on = bool((status_leds & 1 << 8) >> 8)
            pointing_leds_blink = bool((status_leds & 1 << 9) >> 9)
            data_to_store.pointing_leds_blink = pointing_leds_blink
            hv_sector1 = event.slow_data.PDPSlowControl.Sector1_HV
            hv_sector2 = event.slow_data.PDPSlowControl.Sector2_HV
            hv_sector3 = event.slow_data.PDPSlowControl.Sector3_HV
            hv_pdp = np.array(
                np.hstack([hv_sector1, hv_sector2, hv_sector3]), dtype=bool
            )
            data_to_store.all_hv_on = np.all(hv_pdp)
            ghv_sector1 = event.slow_data.PDPSlowControl.Sector1_GHV
            ghv_sector2 = event.slow_data.PDPSlowControl.Sector2_GHV
            ghv_sector3 = event.slow_data.PDPSlowControl.Sector3_GHV
            ghv_pdp = np.array(
                np.hstack([ghv_sector1, ghv_sector2, ghv_sector3]), dtype=bool
            )
            data_to_store.all_ghv_on = np.all(ghv_pdp)
            is_on_source = bool(event.slow_data.DriveSystem.is_on_source)
            data_to_store.is_on_source = is_on_source
            is_tracking = bool(event.slow_data.DriveSystem.is_tracking)
            data_to_store.is_tracking = is_tracking
        for key, val in event.hillas.items():
            data_to_store[key] = val
        output_file.add_container(data_to_store)
    try:
        output_file.close()
        print(hillas_filename, 'created.')
    except ValueError:
        print('WARNING: no data to save,', hillas_filename, 'not created.')
    sys.exit(0)


def entry():
    args = docopt(__doc__)
    files = args['<INPUT>']
    aux_basepath = convert_text(args['--aux_basepath'])
    max_events = convert_int(args['--max_events'])
    dark_filename = args['--dark']
    output = convert_text(args['--output'])
    output_path = os.path.dirname(output)
    if output_path != "" and not os.path.exists(output_path):
        raise IOError('Path ' + output_path +
                      'for output hillas does not exists \n')
    bad_pixels = convert_list_int(args['--bad_pixels'])
    integral_width = convert_int(args['--integral_width'])
    picture_threshold = convert_float(args['--picture_threshold'])
    boundary_threshold = convert_float(args['--boundary_threshold'])
    debug = args['--debug']
    parameters_filename = convert_text(args['--parameters'])
    template_filename = convert_text(args['--template'])
    nevent_plot = convert_int(args['--nevent_plot'])
    event_plot_filename = convert_text(args['--event_plot_filename'])
    disable_bar = args['--disable_bar']
    saturation_threshold = convert_float(args['--saturation_threshold'])
    threshold_pulse = convert_float(args['--threshold_pulse'])
    wdw_number = convert_int(args['--wdw_number'])
    apply_corr_factor = args['--apply_corr_factor']
    if aux_basepath is not None and aux_basepath.lower() == "search":
        input_dir = np.unique([os.path.dirname(file) for file in files])
        if len(input_dir) > 1:
            raise AttributeError(
                "Input files must be from the same directory " +
                "when searching for auxiliaries files"
            )
        input_dir = input_dir[0]
        aux_basepath = input_dir.replace('/raw/', '/aux/')
        if not os.path.isdir(aux_basepath):
            aux_basepath = aux_basepath.replace('/SST1M_01', '/SST1M01')
        if not os.path.isdir(aux_basepath):
            raise AttributeError(
                "Searching for auxiliaries files failed. " +
                "Please use --aux_basepath=PATH"
            )
        print('expecting aux files in', aux_basepath)
    main_pipeline(
        files=files,
        aux_basepath=aux_basepath,
        max_events=max_events,
        dark_filename=dark_filename,
        integral_width=integral_width,
        debug=debug,
        parameters_filename=parameters_filename,
        hillas_filename=output,
        picture_threshold=picture_threshold,
        boundary_threshold=boundary_threshold,
        template_filename=template_filename,
        bad_pixels=bad_pixels,
        disable_bar=disable_bar,
        threshold_pulse=threshold_pulse,
        saturation_threshold=saturation_threshold,
        nevent_plot=nevent_plot,
        event_plot_filename=event_plot_filename,
        wdw_number = wdw_number,
        apply_corr_factor = apply_corr_factor,
    )


if __name__ == '__main__':
    entry()
