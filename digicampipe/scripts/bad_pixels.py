"""
Determine bad pixels from a calibration file
Usage:
    digicam-bad_pixels compute [options]
    digicam-bad_pixels template [options]

Options:
    -h --help                   Show this
    --calib_file=PATH           Calibration YAML file used to determine the bad pixels together with --nsigma_gain and
                                --nsigma_elecnoise [Default: none]
    --parameters=STR            List of parameters to be take into account for the bad pixel tagging. Exact name must be
                                take from the input calibration file. if None, it will iterate among all the keys in the
                                dictionary [Default: None]
    --nsigma=INT                Number of sigmas around the mean value acceptable for the gain. If the gain value of a
                                pixel is outside of the acceptable range, this pixel is considered bad. [Default: 5]
    --nlevels=INT               Number of light levels measured, including the dark level. [Default: 7]
    --dark_histo=FILE           Histogram of the adc samples during dark run used to determine the bad pixels together
                                with --nsigma_dark. [Default: None]
    --nsigma_dark=INT           Number of sigmas around the mean value acceptable for raw adc values in dark. If the
                                value of a pixel is outside of the acceptable range, this pixel is considered bad.
                                [Default: 8]
    --plot=PATH                 path to the output plot. Will show the histograms of the values used to determine bad
                                pixels and highlight those. If set to "show", the plot is displayed and not saved. If
                                set to "none", no plot is done. [Default: show]
    --output=FILE               Same calibration YAML file as the one used as  input plus the list of bad pixels. If
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

from digicampipe.utils.docopt import convert_int, convert_text, convert_list_str


def get_bad_pixels(calib_file=None, keys=None, nsigma=5, nlevels=7,
                   dark_histo=None, nsigma_dark=8,
                   plot="none", output=None):

    bad_pix = np.array([], dtype=int)
    n_pixels = 1296
    if calib_file is not None:
        # Load all the data from the calibration file
        with open(calib_file) as file:
            calib_parameters = yaml.load(file)

    if keys is None:
        keys = []
        for key in calib_parameters:
            keys.append(key)

    for key, value in calib_parameters.items():
        calib_parameters[key] = np.array(value)

    # finding the outlier for each distribution
    sigclip_func = SigmaClip(sigma=nsigma, maxiters=10, cenfunc=np.nanmean, stdfunc=np.nanstd)

    # Array of boolean values for the outlier in each set of parametes
    outlier_array = np.zeros((n_pixels, len(keys)), dtype=int)
    nfailed = np.zeros((n_pixels,), dtype=int)
    parameter_name = np.array([], np.dtype('U100'))
    cnt = 0
    for key, value in calib_parameters.items():
        if key in keys:
            mask_array = sigclip_func(value).mask
            outlier_array[:, cnt] = mask_array
            nfailed += mask_array
            parameter_name = np.append(parameter_name, key)
            cnt += 1

    # masking the good pixels, leaving the bad ones visibles
    full_pixel_table = outlier_array.T
    mask_good_pixels = nfailed > 0
    indx = np.argwhere(mask_good_pixels == True)
    bad_pixel_table = np.zeros((len(keys), len(indx)), dtype=int)

    for i, row in enumerate(full_pixel_table):
        bad_pixel_table[i] = full_pixel_table[i][indx.T]

    full_pixel_table = full_pixel_table.T
    bad_pixel_table = bad_pixel_table.T

    # Sorting in decreasing order, parameters and pixels
    n_failed_parameters = nfailed[indx].T
    n_failed_pixels = indx.T

    ind_sorting_parameters = np.argsort(n_failed_parameters)
    ind_sorting_pixels = np.argsort(n_failed_pixels)

    # Saving table in a CSV file
    parameter_name = np.hstack(['pixel', parameter_name, 'fails'])
    for i, parameter in enumerate(parameter_name):
        if i == 0:
            header = parameter_name[i]
        else:
            header = header + ', {}'.format(parameter)
    print(header)

    table = np.vstack((bad_pixel_table.T, n_failed_parameters))
    table = np.insert(table, 0, n_failed_pixels, axis=0)
    table = table.T

    csv_name = plot + 'bad_pixel_table.csv'
    np.savetxt(csv_name, table, delimiter=',', fmt='%i', header=header)

    print('file saved at {}'.format(csv_name))


def get_bad_pixels_bis(calib_file=None, nsigma=5, nlevels=7,
                   dark_histo=None, nsigma_dark=8,
                   plot="none", output=None):

    bad_pix = np.array([], dtype=int)
    if calib_file is not None:
        # Load all the data from the calibration file
        with open(calib_file) as file:
            calibration_parameters = yaml.load(file)

        gain = np.array(calibration_parameters['gain'])
        elecnoise = np.array(calibration_parameters['sigma_e'])
        baseline = np.array(calibration_parameters['baseline'])
        smearing = np.array(calibration_parameters['sigma_s'])
        crosstalk = np.array(calibration_parameters['xt'])
        dcr = np.array(calibration_parameters['dcr'])

        chi2 = []
        for level in range(nlevels):
            key_name = 'chi2_{}'.format(level)
            chi2.append(calibration_parameters[key_name])
        chi2 = np.array(chi2)

        # Masking the empty pixel (not included in the measurements)
        nan_mask = np.logical_or(~np.isfinite(gain), ~np.isfinite(elecnoise))
        nan_mask = np.logical_or(~nan_mask, ~np.isfinite(baseline))
        nan_mask = np.logical_or(~nan_mask, ~np.isfinite(smearing))
        nan_mask = np.logical_or(~nan_mask, ~np.isfinite(crosstalk))
        nan_mask = np.logical_or(~nan_mask, ~np.isfinite(dcr))

        for level in range(nlevels):
            nan_mask = np.logical_or(~nan_mask, ~np.isfinite(chi2[level]))

        # finding the outlier for each distribution
        sigclip_func = SigmaClip(sigma=nsigma, maxiters=10, cenfunc=np.nanmean, stdfunc=np.nanstd)

        # By applying this mask, I will get only the outlier
        mask_array = []
        gain_mask = sigclip_func(gain).mask
        mask_array.append(gain_mask)
        elecnoise_mask = sigclip_func(elecnoise).mask
        mask_array.append(elecnoise_mask)
        baseline_mask = sigclip_func(baseline).mask
        mask_array.append(baseline_mask)
        smearing_mask = sigclip_func(smearing).mask
        mask_array.append(smearing_mask)
        crosstalk_mask = sigclip_func(crosstalk).mask
        mask_array.append(crosstalk_mask)
        dcr_mask = sigclip_func(dcr).mask
        mask_array.append(dcr_mask)
        chi2_mask = []
        for level in range(nlevels):
            chi2_mask.append(sigclip_func(chi2[level]).mask)
            mask_array.append(chi2_mask)
        chi2_mask = np.array(chi2_mask)


        # Therefore, the pixel where is "True" are the outliers
        gain_outlier = np.argwhere(gain_mask == True)
        elecnoise_outlier = np.argwhere(elecnoise_mask == True)
        baseline_outlier = np.argwhere(baseline_mask == True)
        smearing_outlier = np.argwhere(smearing_mask == True)
        crosstalk_outlier = np.argwhere(crosstalk_mask == True)
        dcr_outlier = np.argwhere(dcr_mask == True)
        chi2_outlier = []
        for level in range(nlevels):
            temp = np.transpose(np.argwhere(chi2_mask[level] == True))
            chi2_outlier.append(temp[0])
        chi2_outlier = np.array(chi2_outlier)

        for i in range(len(chi2_outlier)):
            if i == 0:
                bad_pixels_id = chi2_outlier[i]
            else:
                bad_pixels_id = np.append(bad_pixels_id, chi2_outlier[i])

        bad_pixels_id = np.append(bad_pixels_id, gain_outlier)
        bad_pixels_id = np.append(bad_pixels_id, elecnoise_outlier)
        bad_pixels_id = np.append(bad_pixels_id, baseline_outlier)
        bad_pixels_id = np.append(bad_pixels_id, smearing_outlier)
        bad_pixels_id = np.append(bad_pixels_id, crosstalk_outlier)
        bad_pixels_id = np.append(bad_pixels_id, dcr_outlier)

        bad_pixels_id = np.unique(bad_pixels_id)
        print(bad_pixels_id)

        # Making table with matplotlib
        parameter_names = []
        parameter_names.append('Gain')
        parameter_names.append('Noise')
        parameter_names.append('Baseline')
        parameter_names.append('Smearing')
        parameter_names.append('Crosstalk')
        parameter_names.append('Dark count')
        for level in range(nlevels):
            template_name = 'Template {}'.format(level)
            parameter_names.append(template_name)

        #table = np.zeros((len(bad_pixels_id), len(parameter_names)))

        generic_row = np.arange(len(parameter_names))
        generic_column = np.arange(len(bad_pixels_id))
        table = np.zeros((len(generic_column), len(generic_row)))
        print(table.shape)
        print(table)
        for i, row in enumerate(table):
            if i % 2 == 0:
                table[i, :] = np.zeros((len(generic_row)))
            else:
                table[i, :] = np.ones((len(generic_row)))
        print(table)

        fig, ax = plt.subplots()
        clust_data = table
        collabel = tuple(generic_row)
        rowlabel = tuple(generic_column)
        ax.axis('tight')
        ax.axis('off')
        the_table = ax.table(cellText=clust_data, colLabels=collabel, loc='center')
        plt.show()

        import pandas as pd
        fig, ax = plt.subplots()
        # hide axes
        fig.patch.set_visible(False)
        ax.axis('off')
        ax.axis('tight')
        df = pd.DataFrame(table, columns=collabel)
        ax.table(cellText=df.values, colLabels=df.columns, loc='center')
        fig.tight_layout()
        plt.show()

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
    keys = convert_list_str(args['--parameters'])
    nsigma = convert_int(args['--nsigma'])
    nlevels = convert_int(args['--nlevels'])
    dark_file = convert_text(args['--dark_histo'])
    nsigma_dark = convert_int(args['--nsigma_dark'])
    plot = convert_text(args['--plot'])
    output = convert_text(args['--output'])

    if args['compute']:

        bad_pix = get_bad_pixels(calib_file, keys, nsigma, nlevels, dark_file, nsigma_dark, plot, output)
        print('bad pixels found:', bad_pix)

    if args['template']:
        print('template')


if __name__ == '__main__':
    entry()
