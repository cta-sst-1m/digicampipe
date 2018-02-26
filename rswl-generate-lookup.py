import numpy as np
from optparse import OptionParser
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.stats import binned_statistic
from scipy import interpolate
from shower_geometry import impact_parameter


def plot_lookup2d(data, title):  

    data_width = data[:,[0,1,2]]
    data_length = data[:,[0,1,3]]
    data_sigmaw = data[:,[0,1,4]]
    data_sigmal = data[:,[0,1,5]]
    
    width = data_width[:,2].reshape((len(np.unique(data_width[:,0])), len(np.unique(data_width[:,1]))))
    xw, yw = np.meshgrid(np.unique(data_width[:,1]), np.unique(data_width[:,0]))
    length = data_length[:,2].reshape((len(np.unique(data_length[:,0])), len(np.unique(data_length[:,1]))))
    xl, yl = np.meshgrid(np.unique(data_length[:,1]), np.unique(data_length[:,0]))
    
    sigmaw = data_sigmaw[:,2].reshape((len(np.unique(data_sigmaw[:,0])), len(np.unique(data_sigmaw[:,1]))))
    xsw, ysw = np.meshgrid(np.unique(data_sigmaw[:,1]), np.unique(data_sigmaw[:,0]))
    sigmal = data_sigmal[:,2].reshape((len(np.unique(data_sigmal[:,0])), len(np.unique(data_sigmal[:,1]))))
    xsl, ysl = np.meshgrid(np.unique(data_sigmal[:,1]), np.unique(data_sigmal[:,0]))

    fig, ax = plt.subplots(2, 2,figsize=(14,12))
    
    pcm = ax[0,0].pcolormesh(xl, yl, length, rasterized=True) #, vmin = 0, vmax=1) #, cmap='nipy_spectral')
    ax[0,0].set_ylabel('Impact parameter [m]')
    ax[0,0].set_xlabel('log size')
    cbar = fig.colorbar(pcm, ax=ax[0,0])
    cbar.set_label('mean length')
    
    pcm = ax[0,1].pcolormesh(xsl, ysl, sigmal, rasterized=True) #, vmin = 0, vmax=1) #, cmap='nipy_spectral')
    ax[0,1].set_ylabel('Impact parameter [m]')
    ax[0,1].set_xlabel('log size')
    cbar = fig.colorbar(pcm, ax=ax[0,1])
    cbar.set_label('sigma length')

    pcm = ax[1,0].pcolormesh(xw, yw, width, rasterized=True) #, vmin = 0, vmax=1) #, cmap='nipy_spectral')
    ax[1,0].set_ylabel('Impact parameter [m]')
    ax[1,0].set_xlabel('log size')
    cbar = fig.colorbar(pcm, ax=ax[1,0])
    cbar.set_label('mean width')
    
    pcm = ax[1,1].pcolormesh(xsw, ysw, sigmaw, rasterized=True) #, vmin = 0, vmax=1) #, cmap='nipy_spectral')
    ax[1,1].set_ylabel('Impact parameter [m]')
    ax[1,1].set_xlabel('log size')
    cbar = fig.colorbar(pcm, ax=ax[1,1])
    cbar.set_label('sigma width')


def fill_lookup(impact_bins_edges, size_bins_edges, impact_parameter, size, width, length):

    binned_wls = []
    
    for i in range(len(impact_bins_edges)-1):
        
        imp_edge_min = impact_bins_edges[i]
        imp_edge_max = impact_bins_edges[i+1]
        
        width_impactbinned = width[(impact_parameter >= imp_edge_min) & (impact_parameter < imp_edge_max)]
        length_impactbinned = length[(impact_parameter >= imp_edge_min) & (impact_parameter < imp_edge_max)]

        size_impactbinned = size[(impact_parameter >= imp_edge_min) & (impact_parameter < imp_edge_max)]
        
        for j in range(len(size_bins_edges)-1):
            
            siz_edge_min = size_bins_edges[j]
            siz_edge_max = size_bins_edges[j+1]
            
            if len(width_impactbinned[(size_impactbinned >= siz_edge_min) & (size_impactbinned < siz_edge_max)]) > 0:
                mean_width = np.mean(width_impactbinned[(size_impactbinned >= siz_edge_min) & (size_impactbinned < siz_edge_max)])
                mean_length = np.mean(length_impactbinned[(size_impactbinned >= siz_edge_min) & (size_impactbinned < siz_edge_max)])
                sigma_width = np.std(width_impactbinned[(size_impactbinned >= siz_edge_min) & (size_impactbinned < siz_edge_max)])
                sigma_length = np.std(length_impactbinned[(size_impactbinned >= siz_edge_min) & (size_impactbinned < siz_edge_max)])
            else:
                mean_width = np.nan
                mean_length = np.nan
                sigma_width = np.nan
                sigma_length = np.nan

            binned_wls.append(((imp_edge_max-imp_edge_min)/2.0 + imp_edge_min, (siz_edge_max-siz_edge_min)/2.0 + siz_edge_min, mean_width, mean_length, sigma_width, sigma_length))

    binned_wls = np.array(binned_wls)

    return binned_wls


def rswl(impact_parameter, size, width, length, binned_wls):

    width_lookup = binned_wls[:,[0,1,2]]
    length_lookup = binned_wls[:,[0,1,3]]
    sigmaw_lookup = binned_wls[:,[0,1,4]]
    sigmal_lookup = binned_wls[:,[0,1,5]]

    w = interpolate.griddata((width_lookup[:,0], width_lookup[:,1]), width_lookup[:,2], (impact_parameter, size), method='linear')
    sw = interpolate.griddata((sigmaw_lookup[:,0], sigmaw_lookup[:,1]), sigmaw_lookup[:,2], (impact_parameter, size), method='linear')
    l = interpolate.griddata((length_lookup[:,0], length_lookup[:,1]), length_lookup[:,2], (impact_parameter, size), method='linear')
    sl = interpolate.griddata((sigmal_lookup[:,0], sigmal_lookup[:,1]), sigmal_lookup[:,2], (impact_parameter, size), method='linear')

    rsw = (width - w) / sw
    #rsw = rsw[~np.isnan(rsw)*~np.isinf(rsw)]
    rsl = (length - l) / sl
    #rsl = rsl[~np.isnan(rsl)*~np.isinf(rsl)]

    return rsw, rsl



if __name__ == '__main__':

    parser = OptionParser()
    parser.add_option("-l", "--hilp", dest="hillas_prot", help="path to a file with hillas parameters of protons", default='../../../sst-1m_simulace/data_test/ryzen_testprod/0.0deg/Data//hillas_proton_ze00_az000_p13_b07.npz')
    parser.add_option("-a", "--hilg", dest="hillas_gamma", help="path to a file with hillas parameters of gamma", default='../../../sst-1m_simulace/data_test/ryzen_testprod/0.0deg/Data//hillas_gamma_ze00_az000_p13_b07.npz')
    parser.add_option("-m", "--mcg", dest="mc_gamma", help="path to a file with shower MC parameters of gamma", default='../../../sst-1m_simulace/data_test/ryzen_testprod/0.0deg/Data/shower_param_gamma_ze00_az000.txt')
    parser.add_option("-c", "--mcp", dest="mc_prot", help="path to a file with shower MC parameters of protons", default='../../../sst-1m_simulace/data_test/ryzen_testprod/0.0deg/Data/shower_param_proton_ze00_az000.txt')
    (options, args) = parser.parse_args()

    hillas_prot = np.load(options.hillas_prot)
    mc_prot = np.loadtxt(options.mc_prot)
    hillas_gamma = np.load(options.hillas_gamma)
    mc_gamma = np.loadtxt(options.mc_gamma)

    size_prot = hillas_prot['size']
    border_prot = hillas_prot['border']
    size_gamma = hillas_gamma['size']
    border_gamma = hillas_gamma['border']
    width_gamma = hillas_gamma['width']
    length_gamma = hillas_gamma['length']
    width_prot = hillas_prot['width']
    length_prot = hillas_prot['length']

    """
    fig = plt.figure(figsize=(9, 8))
    plt.hist(np.log(size_gamma))
    fig = plt.figure(figsize=(9, 8))
    plt.hist(np.log(size_prot))
    """

    min_size = 50

    # Masking border flagged events
    mask0_p = [x == 0 for x in border_prot]
    mask1_p = [x > min_size for x in size_prot]
    mask2_p = [~np.isnan(x)*~np.isinf(x) for x in length_prot/width_prot]
    mask0_g = [x == 0 for x in border_gamma]
    mask1_g = [x > min_size for x in size_gamma]
    mask2_g = [~np.isnan(x)*~np.isinf(x) for x in length_gamma/width_gamma]
    mask_p = np.logical_and(mask0_p, mask1_p)*mask2_p
    mask_g = np.logical_and(mask0_g, mask1_g)*mask2_g

    mc_gamma = mc_gamma[mask_g,:]
    mc_prot = mc_prot[mask_p,:]

    x_core_prot = mc_prot[:, 9]
    y_core_prot = mc_prot[:, 10]
    theta_prot = mc_prot[:, 4]
    phi_prot = mc_prot[:, 5]
    x_core_gamma = mc_gamma[:, 9]
    y_core_gamma = mc_gamma[:, 10]
    theta_gamma = mc_gamma[:, 4]
    phi_gamma = mc_gamma[:, 5]
    energy_gamma = mc_gamma[:, 3]
    energy_prot = mc_prot[:, 3]
    core_dist_gamma = mc_gamma[:, 2]
    core_dist_prot = mc_prot[:, 2]


    width_gamma = width_gamma[mask_g]
    length_gamma = length_gamma[mask_g]
    size_gamma = np.log(size_gamma[mask_g])     # log size
    width_prot = width_prot[mask_p]
    length_prot = length_prot[mask_p]
    size_prot = np.log(size_prot[mask_p])       # log size


    # Impact parameter
    impact_parameter_prot = impact_parameter(x_core_prot, y_core_prot, 0, 0, 4, theta_prot, phi_prot)    # not optimal, tel. coordinates should be loaded from somewhere..
    impact_parameter_gamma = impact_parameter(x_core_gamma, y_core_gamma, 0, 0, 4, theta_gamma, phi_gamma)

    # Binning in Impact parameter
    impact_bins_edges = np.linspace(0, 700, 30)
    # Binning in sizes
    size_bins_edges = np.linspace(4, 10, 30)

    # Filling lookup tables [size, impact, value]
    binned_wls_gamma = fill_lookup(impact_bins_edges, size_bins_edges, impact_parameter_gamma, size_gamma, width_gamma, length_gamma)
    #binned_wls_proton = fill_lookup(impact_bins_edges, size_bins_edges, impact_parameter_prot, size_prot, width_prot, length_prot)

    # Reduced scaled width and length
    rswg, rslg = rswl(impact_parameter_gamma, size_gamma, width_gamma, length_gamma, binned_wls_gamma)
    rswp, rslp = rswl(impact_parameter_prot, size_prot, width_prot, length_prot, binned_wls_gamma)

    # removing infs, nans..
    # for saving flags for etienne
    energy_gamma = energy_gamma[~np.isnan(rswg)*~np.isinf(rswg)*~np.isnan(rslg)*~np.isinf(rslg)]
    energy_prot = energy_prot[~np.isnan(rswp)*~np.isinf(rswp)*~np.isnan(rslp)*~np.isinf(rslp)]
    core_dist_gamma = core_dist_gamma[~np.isnan(rswg)*~np.isinf(rswg)*~np.isnan(rslg)*~np.isinf(rslg)]
    core_dist_prot = core_dist_prot[~np.isnan(rswp)*~np.isinf(rswp)*~np.isnan(rslp)*~np.isinf(rslp)]  
    rswg_sel = rswg[~np.isnan(rswg)*~np.isinf(rswg)*~np.isnan(rslg)*~np.isinf(rslg)]
    rslg_sel = rslg[~np.isnan(rswg)*~np.isinf(rswg)*~np.isnan(rslg)*~np.isinf(rslg)]
    rswp_sel = rswp[~np.isnan(rswp)*~np.isinf(rswp)*~np.isnan(rslp)*~np.isinf(rslp)]
    rslp_sel = rslp[~np.isnan(rswp)*~np.isinf(rswp)*~np.isnan(rslp)*~np.isinf(rslp)] 
    rswg = rswg_sel
    rslg = rslg_sel
    rswp = rswp_sel
    rslp = rslp_sel

    # Save the lookup tables
    suffix = 'ze00-az000-offset00.txt'
    path = '../../../sst-1m_simulace/data_test/ryzen_testprod/0.0deg/Data/'
    np.savetxt(path+'rswl-lookup-gamma-'+suffix, binned_wls_gamma, fmt='%.5f')
    #np.savetxt(path+'rswl-lookup-proton-'+suffix, binned_wls_proton, fmt='%.5f')

    # Efficiency vs gamma/hadron cut
    # - ratio between N of events passing the cut and all events
    gamma_cut = np.linspace(-10, 20, 200)
    efficiency_gammaw = []
    efficiency_protonw = []
    efficiency_gammal = []
    efficiency_protonl = []
    for ghc in gamma_cut:
        efficiency_gammaw.append(len(rswg[rswg < ghc])/len(rswg))
        efficiency_protonw.append(len(rswp[rswp < ghc])/len(rswp))
        efficiency_gammal.append(len(rslg[rslg < ghc])/len(rslg))
        efficiency_protonl.append(len(rslp[rslp < ghc])/len(rslp))
    efficiency_gammaw = np.array(efficiency_gammaw)
    efficiency_protonw = np.array(efficiency_protonw)
    efficiency_gammal = np.array(efficiency_gammal)
    efficiency_protonl = np.array(efficiency_protonl)

    # Quality factor
    qualityw = efficiency_gammaw / np.sqrt(efficiency_protonw)
    qualityl = efficiency_gammal / np.sqrt(efficiency_protonl)
    qualityw[np.isinf(qualityw)] = 0
    qualityl[np.isinf(qualityl)] = 0

    print('RSW:', max(qualityw), efficiency_gammaw[qualityw == max(qualityw)], efficiency_protonw[qualityw == max(qualityw)])
    print('RSL:', max(qualityl), efficiency_gammal[qualityl == max(qualityl)], efficiency_protonl[qualityl == max(qualityl)])

    optimal_rsw_cut = gamma_cut[qualityw == max(qualityw)]
    optimal_rsl_cut = gamma_cut[qualityl == max(qualityl)]
    print('Optimal RSW and RSL cuts: ', optimal_rsw_cut, optimal_rsl_cut)

    # Saving positive/false detection flags according to Etienne's wish
    gamma_flag = []
    for i in range(len(rswg)):
        if rswg[i] < optimal_rsw_cut:
            gamma_flag.append(1)
        else: gamma_flag.append(0)
    gamma_flag = np.array(gamma_flag)

    proton_flag = []
    for i in range(len(rswp)):
        if rswp[i] > optimal_rsw_cut:
            proton_flag.append(1)
        else: proton_flag.append(0)
    proton_flag = np.array(proton_flag)

    print(len(gamma_flag[gamma_flag == 1]) / len(gamma_flag), len(proton_flag[proton_flag == 1]) / len(proton_flag))

    print(len(energy_gamma), len(core_dist_gamma), len(gamma_flag), len(rswg))

    output_gamma = np.column_stack((energy_gamma, core_dist_gamma, gamma_flag))
    output_prot = np.column_stack((energy_prot, core_dist_prot, proton_flag))
    np.savetxt('../../../sst-1m_simulace/data_test/ryzen_testprod/0.0deg/Data/rsw-gamma.txt', output_gamma, fmt='%.5f %.5f %d')
    np.savetxt('../../../sst-1m_simulace/data_test/ryzen_testprod/0.0deg/Data/rsw-proton.txt', output_prot, fmt='%.5f %.5f %d')


    # PLOTS

    # Look-up tables
    plot_lookup2d(binned_wls_gamma, 'gamma')
    #plot_lookup2d(binned_wls_proton, 'proton')

    # RSW, RSL
    fig, ax = plt.subplots(1, 2,figsize=(14,6))
    weights_g = np.ones_like(rswg)/float(len(rswg))
    weights_p = np.ones_like(rswp)/float(len(rswp))
    ax[0].hist(rswg, bins=100, weights=weights_g, label='gamma', histtype='step', stacked=True, fill=False, linewidth=4, color='black', range=[-10, 20])
    ax[0].hist(rswp, bins=100, weights=weights_p, label='proton', histtype='step', stacked=True, fill=False, linewidth=4, color='red', range=[-10, 20])
    ax[0].set_xlabel('RSW [sigma]')
    ax[0].set_ylabel('Normalised')
    ax[0].legend()

    weights_g = np.ones_like(rslg)/float(len(rslg))
    weights_p = np.ones_like(rslp)/float(len(rslp))
    ax[1].hist(rslg, bins=100, weights=weights_g, label='gamma', histtype='step', stacked=True, fill=False, linewidth=4, color='black', range=[-10, 20])
    ax[1].hist(rslp, bins=100, weights=weights_p, label='proton', histtype='step', stacked=True, fill=False, linewidth=4, color='red', range=[-10, 20])    
    ax[1].set_xlabel('RSL [sigma]')
    ax[1].set_ylabel('Normalised')    
    ax[1].legend()

    # Efficiency
    fig, ax = plt.subplots(1, 2,figsize=(14,6))
    ax[0].plot(gamma_cut,efficiency_gammaw, 'k.', label='gamma')
    ax[0].plot(gamma_cut,efficiency_protonw, 'r.', label='proton')
    ax[0].set_xlabel('RSW gamma cut [sigma]')
    ax[0].set_ylabel('efficiency')
    ax[0].legend()

    ax[1].plot(gamma_cut,efficiency_gammal, 'k.', label='gamma')
    ax[1].plot(gamma_cut,efficiency_protonl, 'r.', label='proton')
    ax[1].set_xlabel('RSL gamma cut [sigma]')
    ax[1].set_ylabel('efficiency')
    ax[1].legend()

    # Quality factor
    fig, ax = plt.subplots(1, 2,figsize=(14,6))
    ax[0].plot(gamma_cut,qualityw, 'k-')
    ax[0].set_xlabel('RSW gamma cut [sigma]')
    ax[0].set_ylabel('Quality factor')

    ax[1].plot(gamma_cut,qualityl, 'k-')
    ax[1].set_xlabel('RSL gamma cut [sigma]')
    ax[1].set_ylabel('Quality factor')


    plt.show()

