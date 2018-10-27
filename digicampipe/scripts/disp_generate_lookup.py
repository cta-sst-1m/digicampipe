#!/usr/bin/env python

"""

Example:
  ./disp_generate_lookup.py \
  --path=/home/jakub/science/fzu/sst-1m_simulace/data_test/ryzen_test2018/ \
  --equation=5 \
  --outpath=./digicampipe/tests/resources/disp_lookup/ \
  -o 0.0 \
  -o 1.0 \
  -o 3.0

Usage:
  disp_generate_lookup.py (-o <args>)... -u <path> -e <int> -p <path>

Options:

  -h --help     Show this screen.
  -p <path>, --path=<path>  Path to all processed files.
  -e <int>, --equation=<int>  Equation for the DISP parameter fit. [default: 5]
  -o <arg..>, --offset=<arg..>
  -u <path>, --outpath=<path>   Path for saving lookup tables
"""

import os

import numpy as np
import pandas as pd
from docopt import docopt
from lmfit import minimize, Parameters, report_fit

from digicampipe.utils.disp import disp_eval, leak_pixels
from digicampipe.utils.events_image import load_image


def disp_minimize(A, width, length, cog_x, cog_y, x_offset, y_offset,
                  psi, skewness, size, leakage2, method):
    (disp_comp, x_source_comp,
     y_source_comp, residuals) = disp_eval(A, width, length, cog_x,
                                           cog_y, x_offset, y_offset,
                                           psi, skewness, size, leakage2,
                                           method)
    return residuals


def main(all_offsets, path, equation, outpath):
    # Create lists of files
    hillas_files = []
    events_image_files = []
    pipedmc_files = []

    for offset in all_offsets:
        full_path = path + offset + 'deg/Data/processed/'
        all_file_list = os.listdir(full_path)

        hillas_files = hillas_files + [full_path + x for x
                                       in all_file_list if 'hillas_gamma' in x]
        events_image_files = events_image_files + [full_path + x for x
                                                   in all_file_list if
                                                   'events_image_gamma' in x]
        pipedmc_files = pipedmc_files + [full_path + x for x
                                         in all_file_list if
                                         'pipedmc_param_gamma' in x]
        pixel_file = full_path + 'pixels.txt'

    # Make lists of zeniths, offsets, azimuths based on hillas files
    offsets = [x for x in all_offsets]
    zeniths = set()
    azimuths = set()
    for filename in hillas_files:
        zenith_file = filename.split('_')[-2][2:]
        azimuth_file = filename.split('_')[-1].split('.')[0][2:]

        zeniths.add(zenith_file)
        azimuths.add(azimuth_file)
    print(zeniths, azimuths, offsets)

    # Minimize DISP parameters for all input combinations of
    # zenith, azimuth and offset

    lookup = []
    for zenith in zeniths:
        for azimuth in azimuths:
            for offset in offsets:

                print(zenith, azimuth, offset)
                try:
                    hillas_file = [x for x in hillas_files if 'ze'
                                   + zenith in x and 'az' + azimuth
                                   in x and offset + 'deg' in x][0]
                    hillas = np.load(hillas_file)
                except Exception:
                    print('WARNING: There are no input files in the path: '
                          + full_path + ' for the following zenith, azimuth, '
                                        'offset combination: ' + str(
                                            zenith) + ' '
                          + str(azimuth) + ' ' + str(offset))
                    continue
                try:
                    mc_file = [x for x in pipedmc_files if 'ze'
                               + zenith in x and 'az' + azimuth
                               in x and offset + 'deg' in x][0]
                    mc0 = np.loadtxt(mc_file)
                except Exception:
                    print(
                        'ERROR: MC file for hillas file ' +
                        hillas_file +
                        ' wasn\'t found in the same path. '
                        'Run simtel_pipeline again and make sure that MC '
                        'files are successfuly saved.')
                    continue
                try:
                    image_file = [x for x in events_image_files if 'ze'
                                  + zenith in x and 'az'
                                  + azimuth in x and offset
                                  + 'deg' in x][0]
                    pixels, image = load_image(pixel_file, image_file)
                except Exception:
                    print(
                        'ERROR: Events_image file for hillas file ' +
                        hillas_file +
                        ' wasn\'t found in the same path. '
                        'Run simtel_pipeline again and make sure that image '
                        'files are successfuly saved.')
                    continue
                pix_x = pixels[0, :]
                pix_y = pixels[1, :]

                # convert hillas into pandas dataframe
                keys = {'size', 'width', 'length',
                        'psi', 'cen_x', 'cen_y', 'skewness'}
                hillas = dict(zip(keys, (hillas[k] for k in keys)))
                hillas = pd.DataFrame(hillas)

                min_size = 200

                mask1 = hillas['length'] / hillas['width'] > 1e-3
                mask2 = hillas['size'] > min_size

                # Border flaged events are masked for method 1 and 3 only.
                # These events are kept for all others methods, because
                # in these cases DISP = f(...,leakage2)
                if equation == 1 or equation == 3:
                    mask0 = hillas['border'] == 0
                    mask = (~np.isnan(hillas['width'])
                            * ~np.isnan(
                        hillas['cen_x']) * mask0 * mask1 * mask2)
                else:
                    mask = (~np.isnan(hillas['width'])
                            * ~np.isnan(hillas['cen_x']) * mask1 * mask2)

                # hillas
                print('N of events before cuts:, ', hillas.shape[0])
                hillas = hillas[mask]
                print('N of events after cuts: ', hillas.shape[0])

                size = hillas['size'].values
                width = hillas['width'].values
                length = hillas['length'].values
                psi = hillas['psi'].values
                cog_x = hillas['cen_x'].values  # in mm
                cog_y = hillas['cen_y'].values  #
                skewness = hillas['skewness'].values

                # mc
                mc = mc0[mask, :]

                # image
                # there is event number in the first column,
                # the rest are dl1_camera.pe_samples values after cleaning
                image = image[mask, 1:]

                # True MC params
                energy = mc[:, 3]
                x_offset = 0 * np.ones(len(mc[:, 8]))  # source position in deg
                y_offset = -float(offset) * np.ones(len(mc[:, 8]))
                thetap = mc[:, 4]
                phi = mc[:, 5]

                # conversion of coordinates in mm to deg.
                # Digicam: 0.24 deg/px, one SiPM is 24.3 mm wide
                mm_to_deg = 0.24 / 24.3
                cog_x = cog_x * mm_to_deg
                cog_y = cog_y * mm_to_deg

                disp = np.sqrt((x_offset - cog_x) ** 2.0
                               + (y_offset - cog_y) ** 2.0)

                # Outermost pixels
                (leakage2, image_mask,
                 signal_full, signal_border) = leak_pixels(pix_x, pix_y, image)

                # MINIMIZATION OF DISP
                params = Parameters()
                if equation == 1:
                    params.add('A0', value=1.0)
                elif equation == 2:
                    params.add_many(
                        ('A0', 1.0), ('A1', 1.0), ('A2', 1.0),
                        ('A3', 1.0), ('A4', 1.0), ('A5', 1.0),
                        ('A6', 1.0), ('A7', 1.0), ('A8', 1.0))
                elif equation == 3 or equation == 4:
                    params.add_many(('A0', 1.0), ('A1', 1.0))
                elif equation == 5:
                    params.add_many(('A0', 1.0), ('A1', 1.0), ('A2', 1.0))

                out = minimize(disp_minimize, method='leastsq', params=params,
                               args=(
                                   width, length, cog_x, cog_y, x_offset,
                                   y_offset, psi, skewness, size, leakage2,
                                   equation))
                (disp_comp, x_source_comp,
                 y_source_comp, residuals) = disp_eval(out.params, width,
                                                       length, cog_x, cog_y,
                                                       x_offset, y_offset, psi,
                                                       skewness, size,
                                                       leakage2, equation)
                report_fit(out, min_correl=0.1)

                if len(params) > 1:

                    vec = [float(azimuth), float(zenith), float(offset)]
                    for i in range(len(params)):
                        vec += [out.params['A' + str(i)].value,
                                out.params['A' + str(i)].stderr]
                    lookup.append(vec)
                else:
                    lookup.append(
                        [float(azimuth), float(zenith), float(offset),
                         out.params['A0'].value, out.params['A0'].stderr])

    lookup = np.array(lookup)

    if equation == 1:

        np.savetxt(outpath + 'disp_lookup_method1.txt', lookup,
                   fmt='%.6f', header='AZIMUTH  ZENITH  OFFSET  A0  A0_ERR')
        np.savez(outpath + 'disp_lookup_method1',
                 azimuth=lookup[:, 0], zenith=lookup[:, 1],
                 offset=lookup[:, 2], A0=lookup[:, 3],
                 A0_ERR=lookup[:, 4])

    elif equation == 2:

        np.savetxt(outpath + 'disp_lookup_method2.txt', lookup,
                   fmt='%.6f', header=('AZIMUTH  ZENITH  OFFSET  A0  A0_ERR '
                                       'A1  A1_ERR  A2  A2_ERR  A3  A3_ERR '
                                       'A4  A4_ERR A5  A5_ERR  A6  A6_ERR '
                                       'A7  A7_ERR  A8  A8_ERR'))
        np.savez(
            outpath + 'disp_lookup_method5',
            azimuth=lookup[:, 0], zenith=lookup[:, 1], offset=lookup[:, 2],
            A0=lookup[:, 3], A0_ERR=lookup[:, 4],
            A1=lookup[:, 5], A1_ERR=lookup[:, 6],
            A2=lookup[:, 7], A2_ERR=lookup[:, 8],
            A3=lookup[:, 9], A3_ERR=lookup[:, 10],
            A4=lookup[:, 11], A4_ERR=lookup[:, 12],
            A5=lookup[:, 13], A5_ERR=lookup[:, 14],
            A6=lookup[:, 15], A6_ERR=lookup[:, 16],
            A7=lookup[:, 17], A7_ERR=lookup[:, 18],
            A8=lookup[:, 19], A8_ERR=lookup[:, 20])

    elif equation == 3 or equation == 4:

        np.savetxt(
            outpath + 'disp_lookup_method' + str(equation) + '.txt', lookup,
            fmt='%.6f', header=('AZIMUTH  ZENITH  OFFSET '
                                'A0  A0_ERR  A1  A1_ERR'))
        np.savez(
            outpath + 'disp_lookup_method' + str(equation),
            azimuth=lookup[:, 0], zenith=lookup[:, 1], offset=lookup[:, 2],
            A0=lookup[:, 3], A0_ERR=lookup[:, 4],
            A1=lookup[:, 5], A1_ERR=lookup[:, 6])

    elif equation == 5:

        np.savetxt(outpath + 'disp_lookup_method5.txt', lookup,
                   fmt='%.6f', header=('AZIMUTH  ZENITH  OFFSET  A0 '
                                       'A0_ERR  A1  A1_ERR  A2  A2_ERR'))
        np.savez(
            outpath + 'disp_lookup_method5',
            azimuth=lookup[:, 0], zenith=lookup[:, 1], offset=lookup[:, 2],
            A0=lookup[:, 3], A0_ERR=lookup[:, 4],
            A1=lookup[:, 5], A1_ERR=lookup[:, 6],
            A2=lookup[:, 7], A2_ERR=lookup[:, 8])


def entry():
    args = docopt(__doc__)
    main(
        all_offsets=args['--offset'],
        path=args['--path'],
        equation=int(args['--equation']),
        outpath=args['--outpath']
    )


if __name__ == '__main__':
    entry()
