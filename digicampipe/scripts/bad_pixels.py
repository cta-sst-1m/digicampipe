"""
Determine bad pixels from a calibration file
Usage:
    digicam-bad_pixels table [options]
    digicam-bad_pixels histogram [options]
    digicam-bad_pixels template [options]

Options:
    -h --help                   Show this
    -v --debug                  Enter the debug mode.
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
    --output=FILE               Same calibration YAML file as the one used as input plus the list of bad pixels. If
                                "none" no file is created. It is safe to use the same file as the input. [Default: none]

Commands:
    table                       Compute a bad pixel table with its faulty parameters
    template                    Compute templates
    histogram                   Compute histograms from the distribution of all parameters
"""

import matplotlib.pyplot as plt
import numpy as np
from docopt import docopt
import yaml
import seaborn as sns
from astropy.stats import SigmaClip
from histogram.histogram import Histogram1D

from digicampipe.utils.docopt import convert_int, convert_text, convert_list_str
from matplotlib.backends.backend_pdf import PdfPages
from digicampipe.visualization.plot import plot_histo, plot_array_camera


def get_keys(keys, dictionary):
    """

    :param keys:        list of strings denoting the dictionary keys of interest. You should know them in advance
    :param dictionary:  the dictionary loaded from a yml file, in this case the calibration parameters file
    :return:            a list of keys
    """
    if keys is None:
        keys = []
        for key in dictionary:
            keys.append(key)
            print('The keys are : {}'.format(keys))
    else:
        print('The keys are : {}'.format(keys))

    return keys


def get_bad_pixels_full_table(calib_file=None, keys=None, nsigma=5,
                              dark_histo=None, nsigma_dark=8,
                              plot="none", output=None, debug=None):

    n_pixels = np.arange(1296)

    if debug is not None:
        pdf_original = PdfPages(plot + '/original.pdf')
        pdf_computed = PdfPages(plot + '/computed.pdf')

        pdf_original_red = PdfPages(plot + '/original_red.pdf')
        pdf_computed_red = PdfPages(plot + '/computed_red.pdf')

        pdf_median_normalized = PdfPages(plot + '/median_normalized.pdf')

    if calib_file is not None:
        # Load all the data from the calibration file
        with open(calib_file) as file:
            calib_parameters = yaml.load(file)

    keys = get_keys(keys, calib_parameters)

    temp_dict = {}
    for key, value in calib_parameters.items():
        if key in keys:
            temp_dict[key] = np.array(value)
            if debug is not None:
                fig = plot_histo(data=temp_dict[key], x_label=key, bins=100)
                pdf_original.savefig(fig)

                fig, ax = plt.subplots()
                median = np.median(value)
                a_histogram = Histogram1D(np.linspace(np.min(temp_dict[key]), np.max(temp_dict[key]), 100))
                a_histogram.fill(temp_dict[key])
                a_histogram.draw(axis=ax,label=key)
                ax.axvline(x=median, color='tab:orange', label='median')
                pdf_computed.savefig(fig)

    pdf_original.close()
    pdf_computed.close()
    calib_parameters = temp_dict
    del temp_dict

    if ('charge_chi2' in keys) and ('charge_ndf' in keys):
        reduced_chi2 = calib_parameters['charge_chi2'] / calib_parameters['charge_ndf']
        del calib_parameters['charge_chi2']
        del calib_parameters['charge_ndf']
        calib_parameters['red_charge_chi2'] = reduced_chi2

    keys = []
    for key in calib_parameters.keys():
        keys.append(key)
        print(key)

    if debug is not None:
        for key, value in calib_parameters.items():
            if key in keys:
                fig = plot_histo(data=value, x_label=key, bins=100)
                pdf_original_red.savefig(fig)
                plt.close(fig)

                fig, ax = plt.subplots()
                median = np.median(value)
                a_histogram = Histogram1D(np.linspace(np.min(value), np.max(value), 100))
                a_histogram.fill(value)
                a_histogram.draw(axis=ax, label=key)
                ax.axvline(x=median, color='tab:orange', label='median')
                ax.legend(loc=7)
                pdf_computed_red.savefig(fig)
                plt.close(fig)
        pdf_original_red.close()
        pdf_computed_red.close()

    temp_dict = {}
    for key, value in calib_parameters.items():

        # Find the median, make new set of distribution
        median = np.median(value)
        n = 2
        left_bound = median - n * np.abs(median)
        right_bound = median + n * np.abs(median)
        mask = (value >= left_bound) * (value <= right_bound)
        mean = np.mean(value[mask])
        std = np.std(value[mask])

        temp_dict[key] = {}

        temp_dict[key]['value'] = value
        temp_dict[key]['red_value'] = value[mask]
        temp_dict[key]['median'] = np.median(value)
        temp_dict[key]['red_mean'] = mean
        temp_dict[key]['red_std'] = std

        fig, ax = plt.subplots()
        the_values = value[mask]
        a_histogram = Histogram1D(np.linspace(the_values.min(), the_values.max(), 100))
        a_histogram.fill(the_values)
        a_histogram.draw(axis=ax, label=key)
        ax.axvline(x=median, color='tab:orange', label='median')
        ax.axvline(x=mean, color='tab:red', label='mean', linestyle='dashed')
        ax.axvline(x=left_bound, color='tab:orange', linestyle='dashed', label='-{} median'.format(n))
        ax.axvline(x=right_bound, color='tab:orange', linestyle='dashed', label='{} median'.format(n))
        ax.legend(loc='center right')
        pdf_median_normalized.savefig(fig)
        print('for key : {}, median std dist. {} , std from histo {}'.format(key, std, a_histogram.std()))

        del left_bound, right_bound
        plt.close(fig)

    pdf_median_normalized.close()
    calib_parameters = temp_dict

    # Normalize all by the mean or by sigma? TODO
    sigma_normalisation = True
    mean_normalisation = False

    if sigma_normalisation is True:
        normalized_by = 'sigma'
    elif mean_normalisation is True:
        normalized_by = 'mean'
    else:
        normalized_by = 'median'

    if debug:
        pdf_normalized_distribution = PdfPages(plot + '/{}_normalized_distribution.pdf'.format(normalized_by))

    normalized_dist = {}
    for key, value in calib_parameters.items():

        if sigma_normalisation is True:
            normalisation_factor = calib_parameters[key]['red_std']
        elif mean_normalisation is True:
            normalisation_factor = calib_parameters[key]['red_mean']
        else:
            normalisation_factor = calib_parameters[key]['median']

        normalized_dist[key] = {}
        normalized_dist[key]['value'] = calib_parameters[key]['value'] / normalisation_factor
        normalized_dist[key]['normalized_by'] = normalized_by

        if debug:

            fig, ax = plt.subplots()
            the_values = calib_parameters[key]['red_value']
            #the_values = normalized_dist[key]['value']
            a_histogram = Histogram1D(np.linspace(the_values.min(), the_values.max(), 100))
            a_histogram.fill(the_values)
            a_histogram.draw(axis=ax, label=key)
            ax.axvline(x=np.median(the_values), color='tab:orange', label='median')
            ax.axvline(x=np.mean(the_values), color='tab:red', label='mean', linestyle='dashed')
            ax.axvline(x=-np.std(the_values) + np.mean(the_values), color='tab:green', linestyle='dashed', label=r'-$\sigma$')
            ax.axvline(x=np.std(the_values) + np.mean(the_values), color='tab:green', linestyle='dashed', label=r'$\sigma$')
            ax.axvline(x=-2*np.std(the_values) + np.mean(the_values), color='tab:purple', linestyle='dashed', label=r'-2$\sigma$')
            ax.axvline(x=2*np.std(the_values) + np.mean(the_values), color='tab:purple', linestyle='dashed', label=r'2$\sigma$')
            ax.legend(loc='center right')
            pdf_normalized_distribution.savefig(fig)
            plt.close(fig)

    if debug:
        pdf_normalized_distribution.close()

    colormap = 'seismic'

    matrix = np.zeros((len(normalized_dist.keys()), len(n_pixels)))
    # print(keys)
    cnt = 0
    for key, value in normalized_dist.items():
        print(key)
        print(normalized_dist[key]['value'])
        matrix[cnt, :] = normalized_dist[key]['value']
        cnt += 1

    fig, axes = plt.subplots(figsize=(21, 15))

    axes.imshow(matrix, cmap=colormap)
    axes.set_aspect(100)
    plt.savefig('/Users/lonewolf/Desktop/histo_output/table.pdf')
    #plt.show()

    mask2D = np.zeros_like(matrix)
    mask2D[7, 857] = 1
    matrix_masked = np.ma.masked_array(matrix, mask=mask2D)
    axes.imshow(matrix_masked, cmap=colormap)
    axes.set_aspect(100)
    plt.savefig('/Users/lonewolf/Desktop/histo_output/table_0.pdf')

    matrix_masked.mean()
    matrix.mean()



    aa = calib_parameters['xt'].reshape(1, 1296)
    new = np.hsplit(aa, 56)
    fig, ax = plt.subplots()
    ax.imshow(new[0], cmap=colormap)
    ax.set_aspect(100)
    plt.show()

    return 0


def get_bad_pixels_table(calib_file=None, keys=None, nsigma=5,
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

    temp_dict = {}
    for key, value in calib_parameters.items():
        if key in keys:
            temp_dict[key] = np.array(value)
    calib_parameters = temp_dict
    del temp_dict

    # finding the outlier for each distribution
    sigclip_func = SigmaClip(sigma=nsigma, maxiters=10, cenfunc=np.nanmean, stdfunc=np.nanstd)

    # Array of boolean values for the outlier in each set of parameters
    outlier_array = np.zeros((n_pixels, len(keys)), dtype=int)
    nfailed = np.zeros((n_pixels,), dtype=int)
    parameter_name = np.array([], np.dtype('U100'))
    cnt = 0
    for key, value in calib_parameters.items():
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

    return table.shape[0]
    # TODO : bad_pixel array to display here, varaible not used
    # TODO : otion output not use, find a way to improve it


def get_bad_pixels_histograms(calib_file=None, keys=None, nsigma=5,
                              dark_histo=None, nsigma_dark=8,
                              plot="none", output=None):

    bad_pix = np.array([], dtype=int)
    if calib_file is not None:
        with open(calib_file) as file:
            calib_parameters = yaml.load(file)

    if keys is None:
        keys = []
        for key in calib_parameters:
            keys.append(key)

    sigclip_func = SigmaClip(sigma=nsigma, maxiters=10, cenfunc=np.nanmean, stdfunc=np.nanstd)

    temp_dict = {}
    for key, value in calib_parameters.items():
        if key in keys:
            temp_dict[key] = {}
            temp_dict[key]['values'] = np.array(value)
            temp_dict[key]['nan_mask'] = ~np.isfinite(value)
            temp_dict[key]['parameter_mask'] = sigclip_func(value).mask
            temp_dict[key]['bad_mask'] = np.logical_or(~np.isfinite(value), sigclip_func(value).mask)
            temp_dict[key]['bad_pixels'] = np.ndarray.flatten(np.argwhere(temp_dict[key]['bad_mask'] == True).T)
            bad_pix = np.insert(bad_pix, 0, temp_dict[key]['bad_pixels'])
    calib_parameters = temp_dict
    del temp_dict

    bad_pix = np.unique(bad_pix)

    for key, value in calib_parameters.items():
        for sub_key, sub_value in value.items():
            calib_parameters[key][sub_key] = calib_parameters[key][sub_key].tolist()

    if output is not None:
        with open(output, 'w') as file:
            yaml.dump(calib_parameters, file, default_flow_style=True)
    #TODO :  make hisotgram plots, solve the yml problem

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
    calib_file = convert_text(args['--calib_file'])
    debug = args['--debug']
    keys = convert_list_str(args['--parameters'])
    nsigma = convert_int(args['--nsigma'])
    nlevels = convert_int(args['--nlevels'])
    dark_file = convert_text(args['--dark_histo'])
    nsigma_dark = convert_int(args['--nsigma_dark'])
    plot = convert_text(args['--plot'])
    output = convert_text(args['--output'])

    if args['table']:

        #bad_pix = get_bad_pixels_table(calib_file, keys, nsigma, dark_file, nsigma_dark, plot, output)
        #print('bad pixels found:', bad_pix)

        bad_pix = get_bad_pixels_full_table(calib_file, keys, nsigma, dark_file, nsigma_dark, plot, output, debug)
        print('bad pixels found:', bad_pix)

    if args['histogram']:
        # bad_pix = get_bad_pixels_histograms(calib_file, keys, nsigma, nlevels, dark_file, nsigma_dark, plot, output)

        bad_pix = get_bad_pixels_histograms(calib_file=calib_file, keys=keys, nsigma=nsigma,
                                            dark_histo=dark_file, nsigma_dark=nsigma_dark,
                                            plot=plot, output=output)

        print('bad pixels found:', len(bad_pix))


if __name__ == '__main__':
    entry()
