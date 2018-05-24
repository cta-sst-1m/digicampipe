#!/usr/bin/env python

'''

Example:
  ./disp-generate-lookup.py \
  --path=/home/jakub/science/fzu/sst-1m_simulace/data_test/ryzen_test2018/ \
  --equation=4 \
  --outpath=./digicampipe/tests/resources/disp_lookup/ \
  -o 0.0 \
  -o 1.0 \
  -o 3.0

Usage:
  disp-generate-lookup.py (-o <args>)... -u <path> -e <int> -p <path>

Options:

  -h --help     Show this screen.
  -p <path>, --path=<path>  Path to all processed files.
  -e <int>, --equation=<int>  Equation for the DISP parameter fit. [default: 5]
  -o <arg..>, --offset=<arg..>
  -u <path>, --outpath=<path>   Path for saving lookup tables
'''


import numpy as np
from docopt import docopt
import os
import events_image
from lmfit import minimize, Parameters, report_fit
from digicampipe.utils.disp import disp_eval, leak_pixels


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
    zeniths = []
    azimuths = []
    for filename in hillas_files:

        zenith_file = filename.split('_')[-2][2:]
        azimuth_file = filename.split('_')[-1].split('.')[0][2:]

        if zenith_file not in zeniths:
            zeniths.append(zenith_file)
        if azimuth_file not in azimuths:
            azimuths.append(azimuth_file)
    print(zeniths, azimuths, offsets)

    # Minimize DISP parameters for all input combinations of
    # zenith, azimuth and offset

    lookup = []
    for zenith in zeniths:
        for azimuth in azimuths:
            for offset in offsets:

                print(zenith, azimuth, offset)
                hillas = np.load([x for x in hillas_files if 'ze'
                                 + zenith in x and 'az' + azimuth
                                 in x and offset + 'deg' in x][0])
                mc0 = np.loadtxt([x for x in pipedmc_files if 'ze'
                                 + zenith in x and 'az' + azimuth
                                 in x and offset + 'deg' in x][0])
                pixels, image = events_image.load_image(
                                    pixel_file,
                                    [x for x in events_image_files if 'ze'
                                     + zenith in x and 'az'
                                     + azimuth in x and offset
                                     + 'deg' in x][0])
                pix_x = pixels[0, :]
                pix_y = pixels[1, :]

                min_size = 200

                mask1 = [x > 0.001 for x in hillas['width']/hillas['length']]
                mask2 = [x > min_size for x in hillas['size']]

                # Border flaged events are masked for method 1 and 3 only.
                # These events are kept for all others methods, because
                # in these cases DISP = f(...,leakage2)
                if equation == 1 or equation == 3:
                    mask0 = [x == 0 for x in hillas['border']]
                    mask = (~np.isnan(hillas['width'])
                            * ~np.isnan(hillas['cen_x']) * mask0 * mask1 * mask2)
                else:
                    mask = (~np.isnan(hillas['width'])
                            * ~np.isnan(hillas['cen_x']) * mask1 * mask2)

                # hillas
                size = hillas['size'][mask]
                width = hillas['width'][mask]
                length = hillas['length'][mask]
                psi = hillas['psi'][mask]
                cog_x = hillas['cen_x'][mask]  # in mm
                cog_y = hillas['cen_y'][mask]  #
                skewness = hillas['skewness'][mask]

                # mc
                mc = mc0[mask, :]

                # image
                # there is event number in the first column,
                # the rest are dl1_camera.pe_samples values after cleaning
                image = image[mask, 1:]

                print('N of events before cuts:, ', len(hillas['size']))
                print('N of events after cuts: ', len(hillas['size'][mask]))

                # True MC params
                energy = mc[:, 3]
                x_offset = 0*np.ones(len(mc[:, 8]))  # source position in deg
                y_offset = -float(offset)*np.ones(len(mc[:, 8]))
                thetap = mc[:, 4]
                phi = mc[:, 5]

                # conversion of coordinates in mm to deg.
                # Digicam: 0.24 deg/px, one SiPM is 24.3 mm wide
                mm_to_deg = 0.24 / 24.3
                cog_x = cog_x * mm_to_deg
                cog_y = cog_y * mm_to_deg

                disp = np.sqrt((x_offset - cog_x)**2.0
                               + (y_offset - cog_y)**2.0)

                # Outermost pixels
                (leakage2, pix_bound, image_mask,
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
                               args=(width, length, cog_x, cog_y, x_offset,
                               y_offset, psi, skewness, size, leakage2,
                               equation))
                (disp_comp, x_source_comp,
                 y_source_comp, residuals) = disp_eval(out.params, width,
                                                       length, cog_x, cog_y,
                                                       x_offset, y_offset, psi,
                                                       skewness, size, leakage2,
                                                       equation)
                report_fit(out, min_correl=0.1)

                if equation == 1:
                    lookup.append(
                        [float(azimuth), float(zenith),
                         float(offset),
                         out.params['A0'].value, out.params['A0'].stderr
                         ])
                elif equation == 2:
                    lookup.append([float(azimuth), float(zenith),
                        float(offset),
                        out.params['A0'].value, out.params['A0'].stderr,
                        out.params['A1'].value, out.params['A1'].stderr,
                        out.params['A2'].value, out.params['A2'].stderr,
                        out.params['A3'].value, out.params['A3'].stderr,
                        out.params['A4'].value, out.params['A4'].stderr,
                        out.params['A5'].value, out.params['A5'].stderr,
                        out.params['A6'].value, out.params['A6'].stderr,
                        out.params['A7'].value, out.params['A7'].stderr,
                        out.params['A8'].value, out.params['A8'].stderr]
                        )
                elif equation == 3 or equation == 4:
                    lookup.append([float(azimuth), float(zenith),
                        float(offset),
                        out.params['A0'].value, out.params['A0'].stderr,
                        out.params['A1'].value, out.params['A1'].stderr]
                        )
                elif equation == 5:
                    lookup.append([float(azimuth), float(zenith),
                        float(offset),
                        out.params['A0'].value, out.params['A0'].stderr,
                        out.params['A1'].value, out.params['A1'].stderr,
                        out.params['A2'].value, out.params['A2'].stderr]
                        )

    lookup = np.array(lookup)

    if equation == 1:

        np.savetxt(outpath + 'disp_lookup_method1.txt', lookup,
                   fmt='%.6f', header='AZIMUTH  ZENITH  OFFSET  A0  A0_ERR')
        np.savez(options.outpath + 'disp_lookup_method1',
             azimuth=lookup[:, 0], zenith=lookup[:, 1], offset=lookup[:, 2],
             A0=lookup[:, 3], A0_ERR=lookup[:, 4])

    elif equation == 2:

        np.savetxt(outpath + 'disp_lookup_method5.txt', lookup,
                   fmt='%.6f', header='AZIMUTH  ZENITH  OFFSET  A0  A0_ERR  \
                   A1  A1_ERR  A2  A2_ERR  A3  A3_ERR  A4  A4_ERR  \
                   A5  A5_ERR  A6  A6_ERR  A7  A7_ERR  A8  A8_ERR')
        np.savez(options.outpath + 'disp_lookup_method5',
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

        np.savetxt(outpath + 'disp_lookup_method4.txt', lookup,
                   fmt='%.6f', header='AZIMUTH  ZENITH  OFFSET  \
                   A0  A0_ERR  A1  A1_ERR')
        np.savez(outpath + 'disp_lookup_method'+str(equation),
             azimuth=lookup[:, 0], zenith=lookup[:, 1], offset=lookup[:, 2],
             A0=lookup[:, 3], A0_ERR=lookup[:, 4],
             A1=lookup[:, 5], A1_ERR=lookup[:, 6])

    elif equation == 5:

        np.savetxt(outpath + 'disp_lookup_method5.txt', lookup,
                   fmt='%.6f', header='AZIMUTH  ZENITH  OFFSET  A0  \
                   A0_ERR  A1  A1_ERR  A2  A2_ERR')
        np.savez(outpath + 'disp_lookup_method5',
             azimuth=lookup[:, 0], zenith=lookup[:, 1], offset=lookup[:, 2],
             A0=lookup[:, 3], A0_ERR=lookup[:, 4],
             A1=lookup[:, 5], A1_ERR=lookup[:, 6],
             A2=lookup[:, 7], A2_ERR=lookup[:, 8])


if __name__ == '__main__':

    args = docopt(__doc__)
    main(
        all_offsets=args['--offset'],
        path=args['--path'],
        equation=int(args['--equation']),
        outpath=args['--outpath']
    )
