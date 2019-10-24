#!/usr/bin/env python
"""
Do the parameters histogram for all the modules listed in a folder, each module contains 12 pixels and the idea is to extract the fitted parameters (at maximum 1296 pixel) to display

Usage:
  digicam-pdp histograms --output=FILE --ac_levels=INT [--debug --fit_mpe_results=STR --fit_spe_results=STR --n_modules=INT --n_hw_pixel=INT] <INPUT>

Options:
    -h --help                   Show this screen.
    -v --debug                  Enter the debug mode.
    -o --output=FILE            Output file.
                                [default: ./pdp_results]
    --fit_mpe_results=STR       Name of the file with the fit results from multiple photon spectrum (digicampipe-mpe fit combined).
                                [Default: fit_combine.fits]
    --fit_spe_results=STR       Name of the file with the fit from dark count spectra or spe (digicampipe-spe fit).
                                [Default: dc_spe_results.fits]
    --n_modules=INT             Number of measured modules or input modules.
                                [Default: 108]
    --n_hw_pixel=INT            Number of pixel per measured or input module.
                                [Default: 12]
    --ac_levels=INT             number of LED AC DAC levels (this is not a list, unlike other scripts modules)
    Commands:
    histograms                  Compute the histogram
"""

import os

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
from docopt import docopt
from digicampipe.visualization.plot import plot_histo, plot_array_camera
from histogram.histogram import Histogram1D
from iminuit import describe
from tqdm import tqdm
from astropy.table import Table
import fitsio
from scipy.ndimage.filters import convolve1d, convolve


from digicampipe.calib.baseline import fill_digicam_baseline, \
    subtract_baseline
from digicampipe.calib.charge import compute_charge, compute_amplitude
from digicampipe.calib.peak import fill_pulse_indices
from digicampipe.io.event_stream import calibration_event_stream
from digicampipe.utils.docopt import convert_int, \
    convert_pixel_args, convert_list_int
from digicampipe.utils.pdf import mpe_distribution_general
from digicampipe.instrument.light_source import ACLED
from digicampipe.utils.fitter import MPEFitter, FMPEFitter, MPECombinedFitter

import fitsio


def load_camera_config():
    # loading pixels id and coordinates from camera config file
    module_id, pixel_hw_id, pixel_sw_id = np.loadtxt(fname='digicampipe/digicampipe/tests/resources/camera_config.cfg',
                                                     unpack=True,
                                                     skiprows=47,
                                                     usecols=(2, 7, 8))

    return module_id, pixel_hw_id, pixel_sw_id


def get_index(module, hw_pixel):
    """
    Function that yields the correct index for mapping the software pixel from the hardware pixel id list and module id list gicen by the config file
    :param module:                  (int) a module number taken from the folder's name containing the fits files.
                                    for a complete sst-1m camera, values from 1 to 108
    :param hw_pixel:                (int) a pixel hardware pixel number, taken for a loop np.range(1,13)
                                    for a complete module of sst-1m camera, values from 1 to 12
    :return:                        (int) a unique index
    """

    modules = 108
    pixels = 12

    module_id_config, pixel_hw_id_config, pixel_sw_id_config = load_camera_config()

    # Indexes with the same module number in camera config module list
    # This number must be repeated n_pixel times (because n_pixel in each module)
    index_module = np.argwhere(module_id_config == module).reshape((pixels,))

    # Indexes with the same hw pixel number in camera config hw pixel list
    # This number must be repeated n_modules times (because n_modules in each measured batch of modules)
    index_pixel = np.argwhere(pixel_hw_id_config == hw_pixel).reshape((modules,))

    mask = np.isin(element=index_module, test_elements=index_pixel)
    index = index_module[mask]

    if len(index) > 1:
        print('Error, not possible to have 2 indexes for a unique pixel')
        print('Indexes found : {}'.format(index))
        exit()

    return index


def entry():
    args = docopt(__doc__)
    root = args['<INPUT>']
    output = args['--output']
    debug = args['--debug']
    mpe_file = args['--fit_mpe_results']
    spe_file = args['--fit_spe_results']
    n_modules = int(args['--n_modules'])
    n_hw_pixels = int(args['--n_hw_pixel'])
    n_light_levels = int(args['--ac_levels'])

    pixels_in_camera = 1296

    if args['histograms']:

        mpe_baseline = np.zeros((pixels_in_camera,))
        mpe_error_baseline = np.zeros((pixels_in_camera,))
        mpe_gain = np.zeros((pixels_in_camera,))
        mpe_error_gain = np.zeros((pixels_in_camera,))
        mpe_sigma_e = np.zeros((pixels_in_camera,))
        mpe_error_sigma_e = np.zeros((pixels_in_camera,))
        mpe_sigma_s = np.zeros((pixels_in_camera,))
        mpe_error_sigma_s = np.zeros((pixels_in_camera,))
        mpe_mu_xt = np.zeros((pixels_in_camera,))
        mpe_error_mu_xt = np.zeros((pixels_in_camera,))
        mpe_chi_2 = np.zeros((pixels_in_camera,))
        mpe_ndf = np.zeros((pixels_in_camera,))
        mpe_n_peaks = np.zeros((pixels_in_camera,))
        mpe_pixel_ids = np.zeros((pixels_in_camera,))

        mpe_ac_levels = np.zeros((pixels_in_camera, n_light_levels))
        mpe_mean = np.zeros((pixels_in_camera, n_light_levels))
        mpe_std = np.zeros((pixels_in_camera, n_light_levels))
        mpe_mu = np.zeros((pixels_in_camera, n_light_levels))
        mpe_error_mu = np.zeros((pixels_in_camera, n_light_levels))

        spe_dcr = np.zeros((pixels_in_camera,))
        spe_sigma_e = np.zeros((pixels_in_camera,))
        spe_mu_xt = np.zeros((pixels_in_camera,))
        spe_gain = np.zeros((pixels_in_camera,))
        spe_pixels_ids = np.zeros((pixels_in_camera,))

        module_id, pixel_hw_id, pixel_sw_id = load_camera_config()

        if not os.path.exists(output):
            os.makedirs(output)

        dir_list = [int(item) for item in os.listdir(root) if os.path.isdir(os.path.join(root, item))]
        dir_list = np.sort(np.array(dir_list))

        for i, module in enumerate(dir_list):

            mpe = '{}/{}/fits/{}'.format(root, module, mpe_file)
            spe = '{}/{}/fits/{}'.format(root, module, spe_file)
            if debug:
                print('Iteration {} in folder : Module {}'.format(i, module))

            for j, pixel in enumerate(range(1, n_hw_pixels+1)):

                index = get_index(module, pixel)
                sw_index = int(pixel_sw_id[index])

                if debug:
                    print('Iteration {} in module : Pixel {}'.format(j, pixel))
                    print('Index number : {}'.format(index))
                    print('Module number : {}'.format(module_id[index]))
                    print('HW pixel number : {}'.format(pixel_hw_id[index]))
                    print('SW pixel number : {}'.format(pixel_sw_id[index]))
                    print('SW pixel number as int : {}'.format(sw_index))

                with fitsio.FITS(mpe, 'r') as file:

                    if debug:
                        print(file)
                        print(file['MPE_COMBINED'])

                    # Pixel from 0 to 11 to be taken into account per module
                    mpe_baseline[sw_index] = file['MPE_COMBINED']['baseline'].read()[j]
                    mpe_error_baseline[sw_index] = file['MPE_COMBINED']['error_baseline'].read()[j]
                    mpe_gain[sw_index] = file['MPE_COMBINED']['gain'].read()[j]
                    mpe_error_gain[sw_index] = file['MPE_COMBINED']['error_gain'].read()[j]
                    mpe_sigma_e[sw_index] = file['MPE_COMBINED']['sigma_e'].read()[j]
                    mpe_error_sigma_e[sw_index] = file['MPE_COMBINED']['error_sigma_e'].read()[j]
                    mpe_sigma_s[sw_index] = file['MPE_COMBINED']['sigma_s'].read()[j]
                    mpe_error_sigma_s[sw_index] = file['MPE_COMBINED']['error_sigma_s'].read()[j]
                    mpe_mu_xt[sw_index] = file['MPE_COMBINED']['mu_xt'].read()[j]
                    mpe_error_mu_xt[sw_index] = file['MPE_COMBINED']['error_mu_xt'].read()[j]
                    mpe_chi_2[sw_index] = file['MPE_COMBINED']['chi_2'].read()[j]
                    mpe_ndf[sw_index] = file['MPE_COMBINED']['ndf'].read()[j]
                    mpe_n_peaks[sw_index] = file['MPE_COMBINED']['n_peaks'].read()[j]
                    mpe_pixel_ids[sw_index] = file['MPE_COMBINED']['pixel_ids'].read()[j]

                    # An array of size "number of light intensity levels" per pixel
                    mpe_ac_levels[sw_index] = file['MPE_COMBINED']['ac_levels'].read()[j]
                    mpe_mean[sw_index] = file['MPE_COMBINED']['mean'].read()[j]
                    mpe_std[sw_index] = file['MPE_COMBINED']['std'].read()[j]
                    mpe_mu[sw_index] = file['MPE_COMBINED']['mu'].read()[j]
                    mpe_error_mu[sw_index] = file['MPE_COMBINED']['error_mu'].read()[j]

                with fitsio.FITS(spe, 'r') as file:

                    if debug:
                        print(file)
                        print(file['SPE'])

                    # Pixel from 0 to 11 to be taken into account per module
                    spe_dcr[sw_index] = file['SPE']['dcr'].read()[j]
                    spe_sigma_e[sw_index] = file['SPE']['sigma_e'].read()[j]
                    spe_mu_xt[sw_index] = file['SPE']['mu_xt'].read()[j]
                    spe_gain[sw_index] = file['SPE']['gain'].read()[j]
                    spe_pixels_ids[sw_index] = file['SPE']['pixels_ids'].read()[j]

        # Single or unique values of parameters for all light levels
        single_vars = [mpe_baseline,
                       mpe_gain,
                       mpe_sigma_e,
                       mpe_sigma_s,
                       mpe_mu_xt,
                       spe_dcr * 1e3]

        single_vars_names = ['Baseline [LSB]',
                             'Gain [LSB / p.e.]',
                             r'$\sigma_e$ [LSB]',
                             r'$\sigma_s$ [LSB]',
                             r'$\mu_{XT}$ [p.e]',
                             'Dark Count Rate [MHz]']

        # Multipple values of parameters, one per level
        multi_vars = [mpe_mean,
                      mpe_mu]

        multi_vars_names = ['Mean [LSB]',
                            r'$\mu$ [p.e.]']

        pdf = PdfPages('{}/parameters.pdf'.format(output))

        for k, variable in enumerate(single_vars):

            fig = plot_histo(data=variable, x_label=single_vars_names[k], bins='auto')
            pdf.savefig(fig)
            fig.clf()

            cam_display, fig = plot_array_camera(data=variable, label=single_vars_names[k])
            pdf.savefig(fig)
            fig.clf()

        for k, variable in enumerate(multi_vars):
            for l, level in enumerate(range(0, n_light_levels)):

                histogram_label = '{} at light level {}'.format(multi_vars_names[k], l+1)
                fig = plot_histo(data=variable[:, l], x_label=histogram_label, bins='auto')
                pdf.savefig(fig)
                fig.clf()

                cam_display, fig = plot_array_camera(data=variable[:, l], label=histogram_label)
                pdf.savefig(fig)
                fig.clf()

        pdf.close()

    return


if __name__ == '__main__':
    entry()
