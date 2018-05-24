#!/usr/bin/env python

'''

Example:
  ./disp-reconstruct.py \
  --equation=4 \
  --lookup=./digicampipe/tests/resources/disp_lookup/disp_lookup_method4.npz \
  --pixels=/home/jakub/science/fzu/sst-1m_simulace/data_test/ryzen_test2018/0.0deg/Data/processed/pixels.txt \
  --images=/home/jakub/science/fzu/sst-1m_simulace/data_test/ryzen_test2018/0.0deg/Data/processed/events_image_gamma_ze00_az000.txt \
  --modification \
  /home/jakub/science/fzu/sst-1m_simulace/data_test/ryzen_test2018/0.0deg/Data/processed/hillas_gamma_ze00_az000.npz

Usage:
  disp-reconstruct.py [options] <file>...

Options:

  -h --help     Show this screen.
  -e <int>, --equation=<int>  Equation for the DISP parameter fit. [default: 5]
  -l <path>, --lookup=<path>  File with lookup table for selected equation.
  -p <path>, --pixels=<path>  File with pixel coordinates.
  -a <path>, --images=<path>  File with event images.
  --modification    Turn on modified DISP method.
'''

import numpy as np
from docopt import docopt
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import Circle
import events_image
from lmfit import Parameters
from scipy.interpolate import interp2d
from scipy.optimize import curve_fit
from digicampipe.utils.disp import disp_eval, leak_pixels, plot_2d, extents, \
                 arrival_distribution, res_gaussian, r68, r68mod


def main(hillas_file, lookup_file, pixel_file,
         event_image_file, equation, modification):

    hillas = np.load(hillas_file[0])
    lookup = np.load(lookup_file)
    pixels, image = events_image.load_image(pixel_file, event_image_file)
    pix_x = pixels[0, :]
    pix_y = pixels[1, :]

    # just for test purposes, to be readed somehow from real data
    data_offset = 0.0
    data_zenith = 0
    data_azimuth = 0

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

    # image
    # there is event number in the first column, the rest
    # are dl1_camera.pe_samples values that survived cleaning
    image = image[mask, 1:]

    x_offset = 0*np.ones(len(cog_x))    # MC event source position in deg
    y_offset = -float(data_offset)*np.ones(len(cog_x))

    print('N of events before cuts:, ', len(hillas['size']))
    print('N of events after cuts: ', len(hillas['size'][mask]))

    # conversion of coordinates in mm to deg.
    # Digicam: 0.24 deg/px, one SiPM is 24.3 mm wide
    mm_to_deg = 0.24 / 24.3
    cog_x = cog_x * mm_to_deg   # conversion to degrees
    cog_y = cog_y * mm_to_deg   # conversion to degrees

    # Outermost pixels
    (leakage2, pix_bound, image_mask,
     signal_full, signal_border) = leak_pixels(pix_x, pix_y, image)

    # Interpolation in lookup table (need to be done for each
    # azimuth separately or regular grid interpolation have to be implemented)
    A = Parameters()
    if equation == 1:
        f = interp2d(lookup['zenith'],
                     lookup['offset'], lookup['A0'], kind='linear')
        a0 = f(data_zenith, data_offset)
        A.add('A0', value=a0)

    elif equation == 4:
        f1 = interp2d(lookup['zenith'],
                      lookup['offset'], lookup['A0'], kind='linear')
        f2 = interp2d(lookup['zenith'],
                      lookup['offset'], lookup['A1'], kind='linear')
        a0 = f1(data_zenith, data_offset)
        a1 = f2(data_zenith, data_offset)
        A.add_many(('A0', a0), ('A1', a1))

    elif equation == 5:
        f1 = interp2d(lookup['zenith'],
                      lookup['offset'], lookup['A0'], kind='linear')
        f2 = interp2d(lookup['zenith'],
                      lookup['offset'], lookup['A1'], kind='linear')
        f3 = interp2d(lookup['zenith'],
                      lookup['offset'], lookup['A2'], kind='linear')
        a0 = f1(data_zenith, data_offset)
        a1 = f2(data_zenith, data_offset)
        a2 = f3(data_zenith, data_offset)
        A.add_many(('A0', a0), ('A1', a1), ('A2', a2))

    print(A)
    (disp_comp, x_source_comp,
     y_source_comp, residuals) = disp_eval(A, width, length, cog_x, cog_y,
                                           x_offset, y_offset, psi, skewness,
                                           size, leakage2, equation)

    # ARRIVAL DIRECTION DISTRIBUTION
    bins = 100

    if args['--modification']:
        n_triples = 100  # number of randomly chosen triplets for each event
        theta_squared_cut = 0.03    # 3*(maximal_distance)**2
        x_minmax = x_offset[0] + [-1.0, 1.0]
        y_minmax = y_offset[0] + [-1.0, 1.0]
        (n_bin_values, n_bin,
         theta_squared_sum_hist) = arrival_distribution(disp_comp,
                                    x_source_comp, y_source_comp, n_triples,
                                    theta_squared_cut, bins,
                                    x_minmax, y_minmax)

    else:  # needed for 2D gaussian fit
        x_minmax = x_offset[0] + [-1.0, 1.0]
        y_minmax = y_offset[0] + [-1.0, 1.0]
        n_bin = np.histogram2d(x_source_comp, y_source_comp,
                               bins=bins, range=[x_minmax, y_minmax])
        n_bin_values = n_bin[0]

    # RESOLUTION

    # 2D Gaussian fit
    # - creation of a matrix

    # - coordinates of middle of each interval
    x = n_bin[1][:-1] + (n_bin[1][1]-n_bin[1][0])/2.0
    y = n_bin[2][:-1] + (n_bin[2][1]-n_bin[2][0])/2.0
    xx, yy = np.meshgrid(x, y)

    # [x0, y0, sigma, amplitude]
    initial_guess = [x_offset[0], y_offset[0], 0.1, 10, 0]
    gauss_params, uncert_cov = curve_fit(res_gaussian,
                                         (xx.ravel(), yy.ravel()),
                                         n_bin_values.ravel(),
                                         p0=initial_guess)
    perr = np.sqrt(np.diag(uncert_cov))
    print('Sigma[deg] from a fit with 2D gaussian: ',
          abs(gauss_params[2]), '+-', perr[2])

    # Radius of a circle containing 68% of the signal
    res68 = r68(x_source_comp, y_source_comp, x_offset[0], y_offset[0])
    print('R[deg] contains 68% of all events: ', res68[0])
    print('R[deg] contains 68% of events up to R99: ', res68[1])
    print('R[deg] contains 99% (R99): ', res68[2])

    if args['--modification']:

        res68_mod = r68mod(xx.ravel(), yy.ravel(),
                           n_bin_values.ravel(), x_offset[0], y_offset[0])

        print('R68 of MODIFIED DISP: ', res68_mod[0])
        print('R68 of MODIFIED DISP up to R99: ', res68_mod[1])
        print('R99 of MODIFIED DISP: ', res68_mod[2])

    # PLOT RECONSTRUCTED IMAGE

    if args['--modification']:
        fig, ax = plt.subplots(1, 1, figsize=(10, 7))
        ax.imshow(n_bin_values, interpolation='none',
                  extent=extents(x) + extents(y))
        ax.autoscale(False)

        # sigma gauss
        circle = Circle((gauss_params[0], gauss_params[1]),
                        gauss_params[2], facecolor='none',
                        edgecolor="red", linewidth=1, alpha=0.8)
        ax.add_patch(circle)
        ax.scatter(gauss_params[0], gauss_params[1], s=30, c="red",
                   marker="+", linewidth=1)
        ax.set_xlabel('x [deg]')
        ax.set_ylabel('y [deg]')
        plt.tight_layout()

    else:
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))

        ax.hist2d(x_source_comp, y_source_comp, bins=bins,
                  range=np.array([x_minmax, y_minmax]))

        # sigma gauss
        circle = Circle((gauss_params[0], gauss_params[1]), gauss_params[2],
                        facecolor='none', edgecolor="red", linewidth=1,
                        alpha=0.8)
        ax.add_patch(circle)

        # R68
        circle2 = Circle((res68[3], res68[4]), res68[1], facecolor='none',
                         edgecolor="green", linewidth=1, alpha=0.8)
        ax.add_patch(circle2)
        ax.scatter(gauss_params[0], gauss_params[1], s=30, c="red",
                   marker="+", linewidth=1)
        ax.scatter(res68[1], res68[2], s=30, c="green",
                   marker="+", linewidth=1)
        ax.set_xlabel('x [deg]')
        ax.set_ylabel('y [deg]')
        plt.tight_layout()

    plt.show()


if __name__ == '__main__':

    args = docopt(__doc__)
    main(
        hillas_file=args['<file>'],
        lookup_file=args['--lookup'],
        pixel_file=args['--pixels'],
        event_image_file=args['--images'],
        modification=args['--modification'],
        equation=int(args['--equation'])
    )
