from optparse import OptionParser

import matplotlib.pyplot as plt
import numpy as np
import rswl_plot
from fill_lookup import fill_lookup

from jakub.shower_geometry import impact_parameter

if __name__ == '__main__':

    parser = OptionParser()
    parser.add_option('-g', '--hillas', dest='hillas_gamma',
                      help='path to a file with hillas parameters of gamma',
                      default='../../../sst-1m_simulace/data_test/ryzen_testprod/0.0deg/Data/hillas_gamma_ze00_az000_p13_b07.npz')
    parser.add_option('-m', '--mc', dest='mc_gamma',
                      help='path to a file with shower MC parameters of gamma',
                      default='../../../sst-1m_simulace/data_test/ryzen_testprod/0.0deg/Data/shower_param_gamma_ze00_az000.txt')
    parser.add_option('-w', '--output_width', dest='output_width',
                      help='path to an output RSW lookup table',
                      default='../../../sst-1m_simulace/data_test/ryzen_testprod/0.0deg/Data/rsw-lookup-ze00-az000-offset00')
    parser.add_option('-l', '--output_length', dest='output_length',
                      help='path to an output RSL lookup table',
                      default='../../../sst-1m_simulace/data_test/ryzen_testprod/0.0deg/Data/rsl-lookup-ze00-az000-offset00')
    (options, args) = parser.parse_args()

    hillas_gamma = np.load(options.hillas_gamma)
    mc_gamma = np.loadtxt(options.mc_gamma)

    min_size = 50

    # Masking border flagged events
    mask0_g = [x == 0 for x in hillas_gamma['border']]
    mask1_g = [x > min_size for x in hillas_gamma['size']]

    mask_g = np.logical_and(mask0_g, mask1_g)

    # Parameters from MC simulations
    mc_gamma = mc_gamma[mask_g, :]
    x_core_gamma = mc_gamma[:, 9]
    y_core_gamma = mc_gamma[:, 10]
    theta_gamma = mc_gamma[:, 4]
    phi_gamma = mc_gamma[:, 5]

    width_gamma = hillas_gamma['width'][mask_g]
    length_gamma = hillas_gamma['length'][mask_g]
    size_gamma = np.log10(hillas_gamma['size'][mask_g])   # log size

    # Impact parameter
    telpos = np.array([0., 0., 4.])  # not optimal, tel. coordinates should be loaded from somewhere..
    impact_parameter_gamma = impact_parameter(x_core_gamma, y_core_gamma,
                                              telpos, theta_gamma, phi_gamma)

    # Binning in Impact parameter
    impact_bins_edges = np.linspace(0, 600, 30)

    # Binning in size
    size_bins_edges = np.linspace(1.5, 4.5, 30)

    # Filling lookup tables
    binned_rsw = fill_lookup(size_bins_edges, impact_bins_edges,
                             impact_parameter_gamma, size_gamma,
                             width_gamma)

    binned_rsl = fill_lookup(size_bins_edges, impact_bins_edges,
                             impact_parameter_gamma, size_gamma,
                             length_gamma)

    # Save the lookup tables
    np.savez(options.output_width,
             impact=binned_rsw['impact'],
             size=binned_rsw['size'],
             mean=binned_rsw['mean'],
             std=binned_rsw['std'],
             n_data=binned_rsw['n_data'])

    np.savez(options.output_length,
             impact=binned_rsl['impact'],
             size=binned_rsl['size'],
             mean=binned_rsl['mean'],
             std=binned_rsl['std'],
             n_data=binned_rsl['n_data'])

    print('Lookup tables generated and saved..')

    # Plotting lookup tables
    rswl_plot.rswl_lookup2d(binned_rsw, z_axis_title='width')
    rswl_plot.rswl_lookup2d(binned_rsl, z_axis_title='length')
    plt.show()
