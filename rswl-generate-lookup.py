import numpy as np
from optparse import OptionParser
from scipy import interpolate
from shower_geometry import impact_parameter


def fill_lookup(impact_bins_edges, size_bins_edges,
                impact_parameter, size, width, length):

    binned_wls = []

    for i in range(len(impact_bins_edges)-1):

        imp_edge_min = impact_bins_edges[i]
        imp_edge_max = impact_bins_edges[i+1]

        width_impactbinned = width[(impact_parameter >= imp_edge_min) &
                                   (impact_parameter < imp_edge_max)]

        length_impactbinned = length[(impact_parameter >= imp_edge_min) &
                                     (impact_parameter < imp_edge_max)]

        size_impactbinned = size[(impact_parameter >= imp_edge_min) &
                                 (impact_parameter < imp_edge_max)]

        for j in range(len(size_bins_edges)-1):

            siz_edge_min = size_bins_edges[j]
            siz_edge_max = size_bins_edges[j+1]

            if len(width_impactbinned[(size_impactbinned >= siz_edge_min) &
                   (size_impactbinned < siz_edge_max)]) > 0:

                mean_width = np.mean(width_impactbinned[
                    (size_impactbinned >= siz_edge_min) &
                    (size_impactbinned < siz_edge_max)
                    ])
                mean_length = np.mean(length_impactbinned[
                    (size_impactbinned >= siz_edge_min) &
                    (size_impactbinned < siz_edge_max)
                    ])
                sigma_width = np.std(width_impactbinned[
                    (size_impactbinned >= siz_edge_min) &
                    (size_impactbinned < siz_edge_max)
                    ])
                sigma_length = np.std(length_impactbinned[
                    (size_impactbinned >= siz_edge_min) &
                    (size_impactbinned < siz_edge_max)
                    ])
            else:
                mean_width = np.nan
                mean_length = np.nan
                sigma_width = np.nan
                sigma_length = np.nan

            binned_wls.append(
                ((imp_edge_max-imp_edge_min)/2.0 + imp_edge_min,
                (siz_edge_max-siz_edge_min)/2.0 + siz_edge_min,
                mean_width, mean_length, sigma_width, sigma_length
                ))

    binned_wls = np.array(binned_wls)

    return binned_wls


def rswl(impact_parameter, size, width, length, binned_wls):

    width_lookup = binned_wls[:,[0,1,2]]
    length_lookup = binned_wls[:,[0,1,3]]
    sigmaw_lookup = binned_wls[:,[0,1,4]]
    sigmal_lookup = binned_wls[:,[0,1,5]]

    w = interpolate.griddata(
        (width_lookup[:,0], width_lookup[:,1]),
        width_lookup[:,2], (impact_parameter, size), method='linear'
        )

    sw = interpolate.griddata(
        (sigmaw_lookup[:,0], sigmaw_lookup[:,1]),
        sigmaw_lookup[:,2], (impact_parameter, size), method='linear'
        )

    l = interpolate.griddata(
        (length_lookup[:,0], length_lookup[:,1]),
        length_lookup[:,2], (impact_parameter, size),
        method='linear'
        )

    sl = interpolate.griddata(
        (sigmal_lookup[:,0], sigmal_lookup[:,1]),
        sigmal_lookup[:,2], (impact_parameter, size),
        method='linear')

    rsw = (width - w) / sw
    rsw = rsw[~np.isnan(rsw)*~np.isinf(rsw)]
    rsl = (length - l) / sl
    rsl = rsl[~np.isnan(rsl)*~np.isinf(rsl)]

    return rsw, rsl


if __name__ == '__main__':

    parser = OptionParser()
    parser.add_option("-h", "--hillas", dest="hillas_gamma",
        help="path to a file with hillas parameters of gamma",
        default='../../../sst-1m_simulace/data_test/ryzen_testprod/0.0deg/Data/hillas_gamma_ze00_az000_p13_b07.npz')
    parser.add_option("-m", "--mc", dest="mc_gamma",
        help="path to a file with shower MC parameters of gamma",
        default='../../../sst-1m_simulace/data_test/ryzen_testprod/0.0deg/Data/shower_param_gamma_ze00_az000.txt')
    (options, args) = parser.parse_args()

    hillas_gamma = np.load(options.hillas_gamma)
    mc_gamma = np.loadtxt(options.mc_gamma)

    min_size = 50

    # Masking border flagged events
    mask0_g = [x == 0 for x in hillas_gamma['border']]
    mask1_g = [x > min_size for x in hillas_gamma['size']]

    mask_g = np.logical_and(mask0_g, mask1_g)

    # Parameters from MC simulations
    mc_gamma = mc_gamma[mask_g,:]
    x_core_gamma = mc_gamma[:, 9]
    y_core_gamma = mc_gamma[:, 10]
    theta_gamma = mc_gamma[:, 4]
    phi_gamma = mc_gamma[:, 5]

    width_gamma = hillas_gamma['width'][mask_g]
    length_gamma = hillas_gamma['length'][mask_g]
    size_gamma = np.log(hillas_gamma['size'][mask_g])     # log size

    # Impact parameter 
    impact_parameter_gamma = impact_parameter(x_core_gamma, y_core_gamma,
                                              0, 0, 4, theta_gamma, phi_gamma)  # not optimal, tel. coordinates should be loaded from somewhere..

    # Binning in Impact parameter
    impact_bins_edges = np.linspace(0, 700, 30)

    # Binning in size
    size_bins_edges = np.linspace(4, 10, 30)

    # Filling lookup tables [size, impact, value]
    binned_wls = fill_lookup(impact_bins_edges, size_bins_edges,
                             impact_parameter_gamma, size_gamma,
                             width_gamma, length_gamma)

    # Save the lookup tables
    suffix = 'ze00-az000-offset00.txt'
    path = '../../../sst-1m_simulace/data_test/ryzen_testprod/0.0deg/Data/'
    np.savetxt(path+'rswl-lookup-'+suffix, binned_wls, fmt='%.5f')
    print('Lookup table generated and saved..')
