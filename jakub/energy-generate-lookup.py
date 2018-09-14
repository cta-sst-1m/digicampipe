from optparse import OptionParser

import matplotlib.pyplot as plt
import numpy as np
from fill_lookup import fill_lookup
from rswl_plot import energy_lookup2d

from digicampipe.utils.shower_geometry import impact_parameter

if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option(
        "-l",
        "--hillas",
        dest="hillas",
        help="path to a file with hillas parameters",
        default='../../../sst-1m_simulace/data_test/ryzen_testprod/0.0deg/Data/hillas_gamma_ze00_az000_p13_b07.npz')
    parser.add_option(
        "-m",
        "--mc",
        dest="mc",
        help="path to a file with shower MC parameters",
        default='../../../sst-1m_simulace/data_test/ryzen_testprod/0.0deg/Data/shower_param_gamma_ze00_az000.txt')
    parser.add_option(
        '-o',
        '--output',
        dest='output',
        help='path to an output lookup table',
        default='../../../sst-1m_simulace/data_test/ryzen_testprod/0.0deg/Data/energy-lookup-ze00-az000-offset00')
    (options, args) = parser.parse_args()

    hillas = np.load(options.hillas)
    mc = np.loadtxt(options.mc)

    # Masking borderflagged data
    mask = [x == 0 for x in hillas['border']]

    size = hillas['size'][mask]
    mc = mc[mask, :]

    # True MC params
    core_distance = mc[:, 2]
    energy = mc[:, 3]
    x_core = mc[:, 9]
    y_core = mc[:, 10]
    theta = mc[:, 4]
    phi = mc[:, 5]

    # Impact parameter
    # not optimal, tel. coordinates should be loaded from somewhere..
    telpos = np.array([0., 0., 4.])
    impact_parameter = impact_parameter(x_core, y_core, telpos, theta, phi)

    # Binning in log10 size
    size_bins_edges = np.linspace(0.5, 5, 100)

    # Binning in core distance
    impact_bins_edges = np.linspace(0, 500, 100)

    # Filling lookup tables
    binned_energy = fill_lookup(size_bins_edges,
                                impact_bins_edges,
                                impact_parameter,
                                np.log10(size),
                                energy)

    # Save the lookup table
    np.savez(options.output,
             impact=binned_energy['impact'],
             size=binned_energy['size'],
             mean=binned_energy['mean'],
             std=binned_energy['std'],
             n_data=binned_energy['n_data'])

    print('Lookup table generated and saved..')

    # Plotting lookup tables
    energy_lookup2d(binned_energy)
    plt.show()
