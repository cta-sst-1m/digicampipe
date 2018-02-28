import numpy as np
from optparse import OptionParser
import matplotlib.pyplot as plt
import matplotlib as mpl
from shower_geometry import impact_parameter
from rswl_plot import energy_lookup2d


def fill_lookup(size_bins_edges, impact_bins_edges,
                impact_parameter, size, energy):

    binned_energy = []

    for i in range(len(impact_bins_edges)-1):

        imp_edge_min = impact_bins_edges[i]
        imp_edge_max = impact_bins_edges[i+1]

        energy_impactbinned = energy[(impact_parameter >= imp_edge_min) &
                                     (impact_parameter < imp_edge_max)]
        size_impactbinned = size[(impact_parameter >= imp_edge_min) &
                                 (impact_parameter < imp_edge_max)]

        for j in range(len(size_bins_edges)-1):
            
            siz_edge_min = size_bins_edges[j]
            siz_edge_max = size_bins_edges[j+1]
            
            if len(energy_impactbinned[(size_impactbinned >= siz_edge_min) &
                   (size_impactbinned < siz_edge_max)]) > 0:

                mean_energy = np.mean(energy_impactbinned[
                    (size_impactbinned >= siz_edge_min) &
                    (size_impactbinned < siz_edge_max)
                    ])
                std_energy = np.std(energy_impactbinned[
                    (size_impactbinned >= siz_edge_min) &
                    (size_impactbinned < siz_edge_max)
                    ])
                n_energy = len(energy_impactbinned[
                    (size_impactbinned >= siz_edge_min) &
                    (size_impactbinned < siz_edge_max)
                    ])
            else:
                mean_energy = np.nan
                std_energy = np.nan
                n_energy = np.nan

            binned_energy.append(
                ((imp_edge_max-imp_edge_min)/2.0 + imp_edge_min,
                (siz_edge_max-siz_edge_min)/2.0 + siz_edge_min,
                mean_energy, std_energy, n_energy))

    binned_energy = np.array(binned_energy,
                             dtype=[('impact', 'f8'),
                             ('size', 'f8'),
                             ('mean_energy', 'f8'),
                             ('std_energy', 'f8'),
                             ('n_of_events', 'f8')])

    return binned_energy


if __name__ == '__main__':

    parser = OptionParser()
    parser.add_option("-l", "--hillas", dest="hillas",
                      help="path to a file with hillas parameters",
                      default='../../../sst-1m_simulace/data_test/ryzen_testprod/0.0deg/Data/hillas_gamma_ze00_az000_p13_b07.npz')
    parser.add_option("-m", "--mc", dest="mc",
                      help="path to a file with shower MC parameters",
                      default='../../../sst-1m_simulace/data_test/ryzen_testprod/0.0deg/Data/shower_param_gamma_ze00_az000.txt')
    parser.add_option('-o', '--output', dest='output',
                      help='path to an output lookup table',
                      default='../../../sst-1m_simulace/data_test/ryzen_testprod/0.0deg/Data/energy-lookup-ze00-az000-offset00')
    (options, args) = parser.parse_args()

    hillas = np.load(options.hillas) 
    mc = np.loadtxt(options.mc)

    # Masking borderflagged data
    mask = [x == 0 for x in hillas['border']]

    size = hillas['size'][mask]
    mc = mc[mask,:]

    # True MC params
    core_distance = mc[:, 2]
    energy = mc[:, 3]
    x_core = mc[:, 9]
    y_core = mc[:, 10]
    theta = mc[:, 4]
    phi = mc[:, 5]


    # Impact parameter
    telpos = np.array([0., 0., 4.])  # not optimal, tel. coordinates should be loaded from somewhere..
    impact_parameter = impact_parameter(x_core, y_core, telpos, theta, phi)

    # Binning in log10 size
    size_bins_edges = np.linspace(0.5, 5, 100)

    # Binning in core distance
    impact_bins_edges = np.linspace(0, 500, 100)

    # Filling lookup tables [size, impact, value]
    binned_energy = fill_lookup(size_bins_edges,
                                            impact_bins_edges,
                                            impact_parameter,
                                            np.log10(size),
                                            energy)

    # Save the lookup table
    np.savez(options.output,
             impact=binned_energy['impact'],
             size=binned_energy['size'],
             mean_width=binned_energy['mean_energy'],
             mean_length=binned_energy['std_energy'],
             sigma_width=binned_energy['n_of_events'])

    print('Lookup table generated and saved..')

    # Plotting lookup tables
    energy_lookup2d(binned_energy)
    plt.show()
