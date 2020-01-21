"""
Determine bad pixels from a calibration file
Usage:
    digicam-bad_pixels compute [options]
    digicam-bad_pixels template [options]

Options:
    -h --help                   Show this
    --calib_file=PATH           Calibration YAML file used to determine the bad pixels together with --nsigma_gain and
                                --nsigma_elecnoise [Default: none]
    --nsigma_gain=INT           Number of sigmas around the mean value acceptable for the gain. If the gain value of a
                                pixel is outside of the acceptable range, this pixel is considered bad. [Default: 5]
    --nsigma_elecnoise=INT      Number of sigmas around the mean value acceptable for the electronic noise. If the
                                electronic noise value of a pixel is outside of the acceptable range, this pixel is
                                considered bad. [Default: 5]
     --nsigma_baseline=INT      Number of sigmas around the mean value acceptable for the baseline. If the baseline
                                value of a pixel is outside of the acceptable range, this pixel is considered bad.
                                [Default: 5]
     --nsigma_smearing=INT      Number of sigmas around the mean value acceptable for the smearing. If the smearing
                                value of a pixel is outside of the acceptable range, this pixel is considered bad.
                                [Default: 5]
     --nsigma_crosstalk=INT     Number of sigmas around the mean value acceptable for the crosstalk. If the crosstalk
                                value of a pixel is outside of the acceptable range, this pixel is considered bad.
                                [Default: 5]
     --nsigma_dcr=INT           Number of sigmas around the mean value acceptable for the dcr. If the dcr value of a
                                pixel is outside of the acceptable range, this pixel is considered bad. [Default: 5]
     --dark_histo=FILE          Histogram of the adc samples during dark run used to determine the bad pixels together
                                with --nsigma_dark. [Default: None]
     --nsigma_dark=INT          Number of sigmas around the mean value acceptable for raw adc values in dark. If the
                                value of a pixel is outside of the acceptable range, this pixel is considered bad.
                                [Default: 8]
     --plot=PATH                path to the output plot. Will show the histograms of the values used to determine bad
                                pixels and highlight those. If set to "show", the plot is displayed and not saved. If
                                set to "none", no plot is done. [Default: show]
     --output=FILE              Same calibration YAML file as the one used as  input plus the list of bad pixels. If
                                "none" no file is created. It is safe to use the same file as the input. [Default: none]

Commands:
    compute                     Compute the histogram
    template                    Compute templates
"""

import matplotlib.pyplot as plt
import numpy as np
from docopt import docopt
import yaml
import seaborn as sns
from astropy.stats import SigmaClip
from histogram.histogram import Histogram1D

from digicampipe.utils.docopt import convert_int, convert_text


def get_bad_pixels(calib_file=None, nsigma_gain=5, nsigma_elecnoise=5,
                   nsigma_baseline=5, nsigma_smearing=5,
                   nsigma_crosstalk=5, nsigma_dcr=5,
                   dark_histo=None, nsigma_dark=8,
                   plot="none", output=None):

    bad_pix = np.array([], dtype=int)
    if calib_file is not None:
        with open(calib_file) as file:
            calibration_parameters = yaml.load(file)
        gain = np.array(calibration_parameters['gain'])
        elecnoise = np.array(calibration_parameters['sigma_e'])

        baseline = np.array(calibration_parameters['baseline'])
        smearing = np.array(calibration_parameters['sigma_s'])
        crosstalk = np.array(calibration_parameters['xt'])
        dcr = np.array(calibration_parameters['dcr'])

        nan_mask = np.logical_or(~np.isfinite(gain), ~np.isfinite(elecnoise))
        nan_mask = np.logical_or(~nan_mask, ~np.isfinite(baseline))
        nan_mask = np.logical_or(~nan_mask, ~np.isfinite(smearing))
        nan_mask = np.logical_or(~nan_mask, ~np.isfinite(crosstalk))
        nan_mask = np.logical_or(~nan_mask, ~np.isfinite(dcr))

        sigclip_gain = SigmaClip(sigma=nsigma_gain, iters=10,
                                 cenfunc=np.nanmean, stdfunc=np.nanstd)
        sigclip_elecnoise = SigmaClip(sigma=nsigma_elecnoise, iters=10,
                                      cenfunc=np.nanmean, stdfunc=np.nanstd)
        sigclip_baseline = SigmaClip(sigma=nsigma_baseline, iters=10,
                                     cenfunc=np.nanmean, stdfunc=np.nanstd)
        sigclip_smearing = SigmaClip(sigma=nsigma_smearing, iters=10,
                                     cenfunc=np.nanmean, stdfunc=np.nanstd)
        sigclip_crosstalk = SigmaClip(sigma=nsigma_crosstalk, iters=10,
                                      cenfunc=np.nanmean, stdfunc=np.nanstd)
        sigclip_dcr = SigmaClip(sigma=nsigma_dcr, iters=10,
                                cenfunc=np.nanmean, stdfunc=np.nanstd)

        gain_mask = sigclip_gain(gain).mask
        elecnoise_mask = sigclip_elecnoise(elecnoise).mask
        baseline_mask = sigclip_baseline(baseline).mask
        smearing_mask = sigclip_smearing(smearing).mask
        crosstalk_mask = sigclip_crosstalk(crosstalk).mask
        dcr_mask = sigclip_dcr(dcr).mask

        bad_mask = np.logical_or(nan_mask, np.logical_or(gain_mask, elecnoise_mask))
        bad_mask = np.logical_or(bad_mask, np.logical_or(baseline_mask, smearing_mask))
        bad_mask = np.logical_or(bad_mask, np.logical_or(crosstalk_mask, dcr_mask))

        bad_pix = np.where(bad_mask)[0]
        good_pixel_color = sns.xkcd_rgb['cerulean']
        bad_pixel_color = sns.xkcd_rgb['bright red']
        if output is not None:
            calibration_parameters['bad_pixels'] = bad_pix.tolist()
            with open(output, 'w') as file:
                yaml.dump(calibration_parameters, file, default_flow_style=True)
        if plot is not None:
            fig, (ax_gain, ax_elecnoise) = plt.subplots(2, 1)
            gain_bins = np.linspace(np.nanmin(gain), np.nanmax(gain), 100)
            ax_gain.hist(gain[~bad_mask], gain_bins, color=good_pixel_color)
            ax_gain.hist(gain[bad_mask], gain_bins, color=bad_pixel_color)
            ax_gain.set_xlabel('Integral Gain [LSB p.e.$^{-1}$]')
            elecnoise_bins = np.linspace(np.nanmin(elecnoise), np.nanmax(elecnoise), 100)
            ax_elecnoise.hist(elecnoise[~bad_mask], elecnoise_bins, color=good_pixel_color)
            ax_elecnoise.hist(elecnoise[bad_mask], elecnoise_bins, color=bad_pixel_color)
            ax_elecnoise.set_xlabel(r'$\sigma_e$ [LSB]')
            plt.tight_layout()

            if plot != "show":
                plt.savefig(plot + 'gain_enoise.png')
            else:
                plt.show()
            plt.close(fig)

            fig, (ax_baseline, ax_smearing) = plt.subplots(2, 1)
            baseline_bins = np.linspace(np.nanmin(baseline), np.nanmax(baseline), 100)
            ax_baseline.hist(baseline[~bad_mask], baseline_bins, color=good_pixel_color)
            ax_baseline.hist(baseline[bad_mask], baseline_bins, color=bad_pixel_color)
            ax_baseline.set_xlabel('Baseline [LSB]')
            smearing_bins = np.linspace(np.nanmin(smearing), np.nanmax(smearing), 100)
            ax_smearing.hist(smearing[~bad_mask], smearing_bins, color=good_pixel_color)
            ax_smearing.hist(smearing[bad_mask], smearing_bins, color=bad_pixel_color)
            ax_smearing.set_xlabel(r'$\sigma_s$ [LSB]')
            plt.tight_layout()

            if plot != "show":
                plt.savefig(plot + 'baseline_smearing.png')
            else:
                plt.show()
            plt.close(fig)

            fig, (ax_crosstalk, ax_dcr) = plt.subplots(2, 1)
            crosstalk_bins = np.linspace(np.nanmin(crosstalk), np.nanmax(crosstalk), 100)
            ax_crosstalk.hist(crosstalk[~bad_mask], crosstalk_bins, color=good_pixel_color)
            ax_crosstalk.hist(crosstalk[bad_mask], crosstalk_bins, color=bad_pixel_color)
            ax_crosstalk.set_xlabel('Crosstalk [Photons per cell]')
            dcr_bins = np.linspace(np.nanmin(dcr), np.nanmax(dcr), 100)
            ax_dcr.hist(dcr[~bad_mask], dcr_bins, color=good_pixel_color)
            ax_dcr.hist(dcr[bad_mask], dcr_bins, color=bad_pixel_color)
            ax_dcr.set_xlabel('Dark Count Rate [GHz]')
            plt.tight_layout()

            if plot != "show":
                plt.savefig(plot + 'xt_dcr.png')
            else:
                plt.show()
            plt.close(fig)

    if dark_histo is not None:
        dark_histo = Histogram1D.load(dark_histo)
        baseline = dark_histo.mean(method='mid')
        sigclip_dark = SigmaClip(sigma=nsigma_dark, iters=10,
                                 cenfunc=np.nanmean, stdfunc=np.nanstd)
        dark_mask = sigclip_dark(baseline).mask
        bad_dark = np.where(dark_mask)[0]
        bad_pix = np.unique(np.concatenate((bad_pix, bad_dark)))

    print('number of bad pixels : {}'.format(len(bad_pix)))

    return bad_pix


def entry():
    args = docopt(__doc__)
    calib_file = convert_text(args['--calib_file'])
    nsigma_gain = convert_int(args['--nsigma_gain'])
    nsigma_elecnoise = convert_int(args['--nsigma_elecnoise'])
    nsigma_baseline = convert_int(args['--nsigma_baseline'])
    nsigma_smearing = convert_int(args['--nsigma_smearing'])
    nsigma_crosstalk = convert_int(args['--nsigma_crosstalk'])
    nsigma_dcr = convert_int(args['--nsigma_dcr'])
    dark_file = convert_text(args['--dark_histo'])
    nsigma_dark = convert_int(args['--nsigma_dark'])
    plot = convert_text(args['--plot'])
    output = convert_text(args['--output'])

    if args['compute']:
        bad_pix = get_bad_pixels(calib_file, nsigma_gain, nsigma_elecnoise,
                                 nsigma_baseline, nsigma_smearing,
                                 nsigma_crosstalk, nsigma_dcr,
                                 dark_file, nsigma_dark, plot, output)
        print('bad pixels found:', bad_pix)

    if args['template']:
        print('template')


if __name__ == '__main__':
    entry()
