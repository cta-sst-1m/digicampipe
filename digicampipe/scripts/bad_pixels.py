"""
Determine bad pixels from a calibration file.
Usage:
  bad_pixels.py [options] [--] <INPUT>

Options:
  --help                        Show this
  <INPUT>                       Calibration YAML file used to determine the
                                bad pixels. Output of ???
  --nsigma_gain=INT             Number of sigmas around the mean value
                                acceptable for the gain. If the gain value of a
                                pixel is outside of the acceptable range, this
                                pixel is considered bad.
                                [Default: 3]
  --nsigma_elecnoise=INT        Number of sigmas around the mean value
                                acceptable for the electronic noise. If the
                                electronic noise value of a pixel is outside
                                of the acceptable range, this pixel is
                                considered bad.
                                [Default: 3]
  --plot=FILE                   path to the output plot. Will show the
                                histograms of the values used to determine bad
                                pixels and highlight those.
                                If set to "show", the plot is displayed and not
                                saved. If set to "none", no plot is done.
                                [Default: show]
  --output=FILE                 Same calibration YAML file as the one used as
                                input plus the list of bad pixels. If "none"
                                no file is created. It is safe to use the same
                                file as the input.
                                [Default: none]

"""
import matplotlib.pyplot as plt
import numpy as np
from docopt import docopt
import yaml
from astropy.stats import SigmaClip
from histogram.histogram import Histogram1D

from digicampipe.utils.docopt import convert_int, convert_text


def get_bad_pixels(
        calib_file=None, nsigma_gain=5, nsigma_elecnoise=5,
        dark_histo=None, nsigma_dark=8,
        plot="show", output=None
):
    bad_pix = np.array([], dtype=int)
    if calib_file is not None:
        with open(calib_file) as file:
            calibration_parameters = yaml.load(file)
        gain = np.array(calibration_parameters['gain'])
        elecnoise = np.array(calibration_parameters['sigma_e'])
        nan_mask = np.logical_or(~np.isfinite(gain), ~np.isfinite(elecnoise))
        sigclip_gain = SigmaClip(sigma=nsigma_gain, iters=10,
                                 cenfunc=np.nanmean, stdfunc=np.nanstd)
        sigclip_elecnoise = SigmaClip(sigma=nsigma_elecnoise, iters=10,
                                      cenfunc=np.nanmean, stdfunc=np.nanstd)
        gain_mask = sigclip_gain(gain).mask
        elecnoise_mask = sigclip_elecnoise(elecnoise).mask
        bad_mask = np.logical_or(nan_mask,
                                 np.logical_or(gain_mask, elecnoise_mask))
        bad_pix = np.where(bad_mask)[0]
        if output is not None:
            calibration_parameters['bad_pixels'] = bad_pix
            with open(output, 'w') as file:
                yaml.dump(calibration_parameters, file)
        if plot is not None:
            fig, (ax_gain, ax_elecnoise) = plt.subplots(2, 1)
            gain_bins = np.linspace(np.nanmin(gain), np.nanmax(gain), 100)
            ax_gain.hist(gain[~bad_mask], gain_bins, color='b')
            ax_gain.hist(gain[bad_mask], gain_bins, color='r')
            ax_gain.set_xlabel('integral gain [LSB p.e.$^{-1}$]')
            elecnoise_bins = np.linspace(np.nanmin(elecnoise),
                                         np.nanmax(elecnoise), 100)
            ax_elecnoise.hist(elecnoise[~bad_mask], elecnoise_bins, color='b')
            ax_elecnoise.hist(elecnoise[bad_mask], elecnoise_bins, color='r')
            ax_elecnoise.set_xlabel('electronic noise [LSB]')
            plt.tight_layout()
            if plot != "show":
                plt.savefig(plot)
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
    return bad_pix


def entry():
    args = docopt(__doc__)
    calib_file = args['<INPUT>']
    nsigma_gain = convert_int(args['--nsigma_gain'])
    nsigma_elecnoise = convert_int(args['--nsigma_elecnoise'])
    plot = convert_text(args['--plot'])
    output = convert_text(args['--output'])
    bad_pix = get_bad_pixels(calib_file, nsigma_gain, nsigma_elecnoise,
                             plot, output)
    print('bad pixels found:', bad_pix)


if __name__ == '__main__':
    entry()
