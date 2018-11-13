#!/usr/bin/env python
"""
measure the time offsets using template fitting. Create a file
time_acXXX_dcYYY.npz for each AC and DC level with XXX and YYY the value of
resp. the AC and DC level.
Usage:
  digicam-time-resolution [options] [--] <INPUT>...

Options:
  -h --help                     Show this screen.
  <INPUT>                       List of path to zfits files to analyse.
  --ac_levels=LIST              Comma separated list of AC levels for each
                                input file. Must be of the same size as the
                                number of input file.
  --dc_levels=LIST              Comma separated list of DC levels for each
                                input file. Must be a single value or the same
                                size as the number of input files. [default: 0]
  --max_events=INT              Maximum number of events to analyze. Set to
                                none to use all events in the input files.
                                [default: none]
  --delay_step_ns=FLOAT         Unit of delay in ns used to shift the template
                                [default: 0.1]
  --time_range_ns=<RANGE>       Coma separated interval in ns for the pulse
                                template.
                                [default: -9.,39.]
  --normalize_range=<RANGE>     index of the border samples around the max used
                                for normalization. If set to 0,0 the
                                normalization is by the amplitude of the max.
                                [default: -3,4]
  --parameters=FILE             Calibration parameters file path. If set to
                                none, the default one is used.
                                [default: none]
  --template=FILE               Pulse template file path. If set to none, the
                                default one is used.
                                [default: none]
  --output=DIR                  Path to a directory where results will be
                                stored.
                                [default: ./]
"""
from docopt import docopt
from pkg_resources import resource_filename
import os
import numpy as np
import yaml

from digicampipe.utils.docopt import convert_int, convert_float, convert_text,\
    convert_list_float, convert_list_int
from digicampipe.io.event_stream import calibration_event_stream
from digicampipe.calib.baseline import fill_digicam_baseline, subtract_baseline
from digicampipe.utils.pulse_template import NormalizedPulseTemplate


parameters_default = resource_filename(
    'digicampipe',
    os.path.join(
        'tests',
        'resources',
        'calibration_20180814.yml'
    )
)
template_default = resource_filename(
    'digicampipe',
    os.path.join(
        'tests',
        'resources',
        'pulse_template_all_pixels.txt'
    )
)


def analyse_ACDC_level(
    files, max_events, delay_step_ns, time_range_ns, sampling_ns,
    normalize_range, parameters, template, adc_noise
):
    with open(parameters) as parameters_file:
        calibration_parameters = yaml.load(parameters_file)
    gain_pixels = np.array(calibration_parameters['gain'])
    normalize_slice = np.arange(normalize_range[0], normalize_range[1]+1,
                                dtype=int)
    sample_template = np.arange(time_range_ns[0], time_range_ns[1],
                                sampling_ns)
    n_sample_template = len(sample_template)
    template = NormalizedPulseTemplate.load(template)
    delays = np.arange(-4, 4, delay_step_ns)
    n_delays = len(delays)
    templates_ampl = np.zeros([n_delays, n_sample_template])
    templates_std = np.zeros([n_delays, n_sample_template])
    index_max_template = np.zeros(n_delays, dtype=int)
    for i, delay in enumerate(delays):
        ampl_templ = template(sample_template + delay)
        index_max_template[i] = np.argmax(ampl_templ)
        range_integ = index_max_template[i] + normalize_slice
        norm_templ = np.sum(ampl_templ[range_integ])
        templates_ampl[i, :] = ampl_templ / norm_templ
        std = template.std(sample_template + delay) / norm_templ
        templates_std[i, :] = std
    max_ampl_one_pe = np.max(templates_ampl)
    events = calibration_event_stream(files, max_events=max_events,
                                      disable_bar=True)
    events = fill_digicam_baseline(events)
    events = subtract_baseline(events)
    rows_norm = np.tile(
        np.arange(1296, dtype=int)[:, None],
        [1, len(normalize_slice)]
    )
    samples_events = None
    t_fit = []
    charge = []
    for event in events:
        if samples_events is None:
            n_sample = event.data.adc_samples.shape[1]
            samples_events = np.arange(n_sample) * sampling_ns
        adc_samples = event.data.adc_samples
        idx_sample_max = np.argmax(adc_samples, axis=1)
        column_norm = normalize_slice[None, :] + idx_sample_max[:, None]
        # we skip pixels with max too close to the limit of the sampling window
        # to be sure to be able to integrate and get normalization
        good_pix = np.logical_and(
            np.all(column_norm < n_sample, axis=1),
            np.all(column_norm >= 0, axis=1)
        )
        # we skip pixels with max too close to the limit of the sampling window
        # to be sure to be able to compare with full template
        mean_index_max_template = int(np.round(np.mean(index_max_template)))
        index_template_rel = idx_sample_max - mean_index_max_template
        good_pix = np.logical_and(
            good_pix,
            index_template_rel >= 0
        )
        good_pix = np.logical_and(
            good_pix,
            index_template_rel + n_sample_template - 1 < n_sample
        )
        # we discard pixels with less that few pe
        good_pix = np.logical_and(
            good_pix,
            np.max(adc_samples, axis=1) > 3.5 / max_ampl_one_pe
        )
        # discard pixels with max pulse not around the right position
        good_pix = np.logical_and(
            good_pix,
            idx_sample_max >= 15
        )
        good_pix = np.logical_and(
            good_pix,
            idx_sample_max <= 16
        )
        sample_norm = adc_samples[rows_norm[good_pix, :],
                                  column_norm[good_pix, :]]
        norm_pixels = np.sum(sample_norm, axis=1)
        # discard pixels where charge is <= 2.5 LSB (0.5 pe), as normalization
        # is then meaningless
        norm_all = np.zeros(1296)
        norm_all[good_pix] = norm_pixels
        good_pix = np.logical_and(
            good_pix,
            norm_all > 2.5
        )
        norm_pixels = norm_pixels[norm_pixels > 2.5]

        charge_good_pix = norm_pixels / gain_pixels[good_pix]
        adc_samples_norm = adc_samples[good_pix, :] / norm_pixels[:, None]
        n_good_pix = int(np.sum(good_pix))
        samples = np.arange(n_sample_template, dtype=int)[None, None, :]
        column_chi2 = index_template_rel[good_pix, None, None] + samples
        row_chi2 = np.tile(
            np.arange(n_good_pix)[:, None, None],
            [1, n_delays, n_sample_template]
        )
        adc_samples_compared = adc_samples_norm[row_chi2, column_chi2]
        residual = adc_samples_compared - templates_ampl[None, :, :]
        error_squared = templates_std[None, :, :] ** 2 \
            + (adc_noise/norm_pixels[:, None, None]) ** 2
        chi2 = np.sum(residual**2 / error_squared, axis=2) \
            / (n_sample_template - 1)

        t_fit_all = np.ones(1296) * np.nan
        # estimate offset from min chi2
        idx_delay_min = np.argmin(chi2, axis=1)
        delays_min = delays[idx_delay_min]
        delays_min[chi2[np.arange(n_good_pix), idx_delay_min] > 20] = np.nan
        t_fit_all[good_pix] = delays_min

        t_fit.append(-t_fit_all + idx_sample_max * sampling_ns)
        charge_all = np.ones(1296) * np.nan
        charge_all[good_pix] = charge_good_pix
        charge.append(charge_all)
    t_fit = np.array(t_fit)
    charge = np.array(charge)
    return charge, t_fit


def main(
        files, ac_levels, dc_levels, max_events, delay_step_ns, time_range_ns,
        sampling_ns, normalize_range, parameters, template, adc_noise, output
):
    unique_ac_dc, inverse = np.unique(
        np.vstack([ac_levels, dc_levels]).T,
        axis=0,
        return_inverse=True
    )
    files = np.array(files)
    for i, (ac_level, dc_level) in enumerate(unique_ac_dc):
        files_level = files[inverse == i]
        print('analyze file with AC DAC =', ac_level, 'DC DAC =', dc_level)
        charge, t_fit = analyse_ACDC_level(
            files_level, max_events, delay_step_ns, time_range_ns, sampling_ns,
            normalize_range, parameters, template, adc_noise
        )
        filename = os.path.join(
            output,
            'time_ac{}_dc{}.npz'.format(ac_level, dc_level)
        )
        mean_charge = np.nanmean(charge, axis=0)
        std_charge = np.nanstd(charge, axis=0)
        mean_t = np.nanmean(t_fit, axis=0)
        std_t = np.nanstd(t_fit, axis=0)
        np.savez(
            filename,
            mean_charge=mean_charge,
            std_charge=std_charge,
            mean_t=mean_t,
            std_t=std_t,
            ac_level=ac_level,
            dc_level=dc_level
        )


def entry():
    args = docopt(__doc__)
    files = args['<INPUT>']
    ac_levels = convert_list_int(args['--ac_levels'])
    dc_levels = convert_list_int(args['--dc_levels'])
    if len(dc_levels) == 1:
        dc_levels = [dc_levels[0], ] * len(ac_levels)
    assert len(ac_levels) == len(files)
    max_events = convert_int(args['--max_events'])
    delay_step_ns = convert_float(args['--delay_step_ns'])
    time_range_ns = convert_list_float(args['--time_range_ns'])
    normalize_range = convert_list_int(args['--normalize_range'])
    sampling_ns = 4
    parameters = convert_text(args['--parameters'])
    template = convert_text(args['--template'])
    output = convert_text(args['--output'])
    if parameters is None:
        parameters = parameters_default
    if template is None:
        template = template_default
    main(
        files=files,
        ac_levels=ac_levels,
        dc_levels=dc_levels,
        max_events=max_events,
        delay_step_ns=delay_step_ns,
        time_range_ns=time_range_ns,
        sampling_ns=sampling_ns,
        normalize_range=normalize_range,
        parameters=parameters,
        template=template,
        adc_noise=1.,
        output=output
    )


if __name__ == '__main__':
    entry()
