import numpy as np
from optparse import OptionParser
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.stats import binned_statistic
from scipy import interpolate
from shower_geometry import impact_parameter
import rswl_plot


def rswl(impact_parameter, size, width, length, binned_wls):

    width_lookup = binned_wls[:, [0, 1, 2]]
    length_lookup = binned_wls[:, [0, 1, 3]]
    sigmaw_lookup = binned_wls[:, [0, 1, 4]]
    sigmal_lookup = binned_wls[:, [0, 1, 5]]

    wi = interpolate.griddata(
         (width_lookup[:, 0], width_lookup[:, 1]),
         width_lookup[:, 2], (impact_parameter, size), method='linear'
         )

    sw = interpolate.griddata(
        (sigmaw_lookup[:, 0], sigmaw_lookup[:, 1]),
        sigmaw_lookup[:, 2], (impact_parameter, size), method='linear'
        )

    le = interpolate.griddata(
         (length_lookup[:, 0], length_lookup[:, 1]),
         length_lookup[:, 2], (impact_parameter, size),
         method='linear'
         )

    sl = interpolate.griddata(
        (sigmal_lookup[:, 0], sigmal_lookup[:, 1]),
        sigmal_lookup[:, 2], (impact_parameter, size),
        method='linear')

    rsw = (width - wi) / sw
    rsw = rsw[~np.isnan(rsw)*~np.isinf(rsw)]
    rsl = (length - le) / sl
    rsl = rsl[~np.isnan(rsl)*~np.isinf(rsl)]

    return rsw, rsl


def efficiency_comp(gamma_cut, rswl):

    efficiency = []

    for ghc in gamma_cut:

        efficiency.append(len(rswl[rswl < ghc])/len(rswl))

    efficiency = np.array(efficiency)

    return efficiency


if __name__ == '__main__':

    parser = OptionParser()
    parser.add_option('-p', '--hillasp', dest='hillas_prot',
                      help='path to a file with hillas parameters of protons',
                      default='../../../sst-1m_simulace/data_test/ryzen_testprod/0.0deg/Data//hillas_proton_ze00_az000_p13_b07.npz')
    parser.add_option('-g', '--hillasg', dest='hillas_gamma',
                      help='path to a file with hillas parameters of gamma',
                      default='../../../sst-1m_simulace/data_test/ryzen_testprod/0.0deg/Data//hillas_gamma_ze00_az000_p13_b07.npz')
    parser.add_option('-m', '--mcg', dest='mc_gamma',
                      help='path to a file with shower MC parameters of gamma',
                      default='../../../sst-1m_simulace/data_test/ryzen_testprod/0.0deg/Data/shower_param_gamma_ze00_az000.txt')
    parser.add_option('-c', '--mcp', dest='mc_prot',
                      help='path to a file with shower MC param. of protons',
                      default='../../../sst-1m_simulace/data_test/ryzen_testprod/0.0deg/Data/shower_param_proton_ze00_az000.txt')
    parser.add_option('-l', '--lookup', dest='lookup',
                      help='path to a file with lookup table',
                      default='../../../sst-1m_simulace/data_test/ryzen_testprod/0.0deg/Data/rswl-lookup-ze00-az000-offset00.txt')
    (options, args) = parser.parse_args()

    hillas_prot = np.load(options.hillas_prot)
    mc_prot = np.loadtxt(options.mc_prot)
    hillas_gamma = np.load(options.hillas_gamma)
    mc_gamma = np.loadtxt(options.mc_gamma)
    binned_wls = np.loadtxt(options.lookup)

    min_size = 50

    # Masking border flagged events
    mask0_p = [x == 0 for x in hillas_prot['border']]
    mask1_p = [x > min_size for x in hillas_prot['size']]
    mask0_g = [x == 0 for x in hillas_gamma['border']]
    mask1_g = [x > min_size for x in hillas_gamma['size']]

    mask_p = np.logical_and(mask0_p, mask1_p)
    mask_g = np.logical_and(mask0_g, mask1_g)

    mc_gamma = mc_gamma[mask_g, :]
    mc_prot = mc_prot[mask_p, :]

    x_core_prot = mc_prot[:, 9]
    y_core_prot = mc_prot[:, 10]
    theta_prot = mc_prot[:, 4]
    phi_prot = mc_prot[:, 5]
    x_core_gamma = mc_gamma[:, 9]
    y_core_gamma = mc_gamma[:, 10]
    theta_gamma = mc_gamma[:, 4]
    phi_gamma = mc_gamma[:, 5]

    width_gamma = hillas_gamma['width'][mask_g]
    length_gamma = hillas_gamma['length'][mask_g]
    size_gamma = np.log10(hillas_gamma['size'][mask_g])     # log size
    width_prot = hillas_prot['width'][mask_p]
    length_prot = hillas_prot['length'][mask_p]
    size_prot = np.log10(hillas_prot['size'][mask_p])       # log size

    # Impact parameter
    impact_parameter_prot = impact_parameter(x_core_prot, y_core_prot,
                                             0, 0, 4, theta_prot, phi_prot)    # not optimal, tel. coordinates should be loaded from somewhere..
    impact_parameter_gamma = impact_parameter(x_core_gamma, y_core_gamma,
                                             0, 0, 4, theta_gamma, phi_gamma)

    # Reduced scaled width and length
    rswg, rslg = rswl(impact_parameter_gamma,
                      size_gamma, width_gamma,
                      length_gamma, binned_wls)
    rswp, rslp = rswl(impact_parameter_prot,
                      size_prot, width_prot,
                      length_prot, binned_wls)

    # Efficiency vs gamma/hadron cut
    # - ratio between N of events passing the cut and all events
    gamma_cut = np.linspace(-10, 20, 100)
    efficiency_gammaw = efficiency_comp(gamma_cut, rswg)
    efficiency_gammal = efficiency_comp(gamma_cut, rslg)
    efficiency_protonw = efficiency_comp(gamma_cut, rswp)
    efficiency_protonl = efficiency_comp(gamma_cut, rslp)

    # Quality factor
    qualityw = efficiency_gammaw / np.sqrt(efficiency_protonw)
    qualityl = efficiency_gammal / np.sqrt(efficiency_protonl)
    qualityw[np.isinf(qualityw)+np.isnan(qualityw)] = 0
    qualityl[np.isinf(qualityl)+np.isnan(qualityl)] = 0

    print('\n     Quality factor     Efficiency gamma   Efficiency proton')
    print('RSW:', max(qualityw),
          efficiency_gammaw[qualityw == max(qualityw)][0],
          efficiency_protonw[qualityw == max(qualityw)][0])
    print('RSL:', max(qualityl),
          efficiency_gammal[qualityl == max(qualityl)][0],
          efficiency_protonl[qualityl == max(qualityl)][0])
    print('')
    print('                 Width               Length')
    print('Optimal cuts:',
          gamma_cut[qualityw == max(qualityw)][0],
          gamma_cut[qualityl == max(qualityl)][0])

    # PLOTS

    # Look-up tables
    rswl_plot.lookup2d(binned_wls)

    # RSW, RSL
    rswl_plot.rswl_norm(rswg, rslg, rswp, rslp)

    # Efficiency
    rswl_plot.efficiency(gamma_cut, efficiency_gammaw, efficiency_gammal,
                         efficiency_protonw, efficiency_protonl)

    # Quality factor
    rswl_plot.quality(gamma_cut, qualityw, qualityl)

    plt.show()
