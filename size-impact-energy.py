import numpy as np
from optparse import OptionParser
import matplotlib.pyplot as plt
import matplotlib as mpl
from shower_geometry import impact_parameter


def fill_lookup(size_bins_edges, impact_bins_edges, impact_parameter, size, energy):

    binned_energy = []
    
    for i in range(len(impact_bins_edges)-1):
        
        imp_edge_min = impact_bins_edges[i]
        imp_edge_max = impact_bins_edges[i+1]
        
        energy_impactbinned = energy[(impact_parameter >= imp_edge_min) & (impact_parameter < imp_edge_max)]
        size_impactbinned = size[(impact_parameter >= imp_edge_min) & (impact_parameter < imp_edge_max)]

        for j in range(len(size_bins_edges)-1):
            
            siz_edge_min = size_bins_edges[j]
            siz_edge_max = size_bins_edges[j+1]
            
            if len(energy_impactbinned[(size_impactbinned >= siz_edge_min) & (size_impactbinned < siz_edge_max)]) > 0:
                mean_energy = np.mean(energy_impactbinned[(size_impactbinned >= siz_edge_min) & (size_impactbinned < siz_edge_max)])
                std_energy = np.std(energy_impactbinned[(size_impactbinned >= siz_edge_min) & (size_impactbinned < siz_edge_max)])
                n_energy = len(energy_impactbinned[(size_impactbinned >= siz_edge_min) & (size_impactbinned < siz_edge_max)])
            else:
                mean_energy = np.nan
                std_energy = np.nan
                n_energy = np.nan

            binned_energy.append(((imp_edge_max-imp_edge_min)/2.0 + imp_edge_min, (siz_edge_max-siz_edge_min)/2.0 + siz_edge_min, mean_energy, std_energy, n_energy))

    binned_energy = np.array(binned_energy)

    return binned_energy


def plot_lookup2d(data):  

    
    energy = data[:,2].reshape((len(np.unique(data[:,0])), len(np.unique(data[:,1]))))
    xe, ye = np.meshgrid(np.unique(data[:,1]), np.unique(data[:,0]))
    
    sigmae = data[:,3].reshape((len(np.unique(data[:,0])), len(np.unique(data[:,1]))))
    xse, yse = np.meshgrid(np.unique(data[:,1]), np.unique(data[:,0]))

    n_energy = data[:,4].reshape((len(np.unique(data[:,0])), len(np.unique(data[:,1]))))
    xn, yn = np.meshgrid(np.unique(data[:,1]), np.unique(data[:,0]))

    fig, ax = plt.subplots(1,figsize=(10,8))
    pcm = ax.pcolormesh(ye, xe, np.log10(energy), rasterized=True) #, vmin = 0, vmax=1) #, cmap='nipy_spectral')
    ax.set_xlabel('Impact parameter [m]')
    ax.set_ylabel('log10(size) [phot]')
    cbar = fig.colorbar(pcm, ax=ax)
    ax.set_xlim([0, 500])
    ax.set_ylim([1.5, 4.5])
    cbar.set_label('log10(E) [TeV]')
    
    fig, ax = plt.subplots(1,figsize=(10,8))
    pcm = ax.pcolormesh(ye, xe, sigmae/energy, rasterized=True) #, vmin = 0, vmax=1) #, cmap='nipy_spectral')
    ax.set_xlabel('Impact parameter [m]')
    ax.set_ylabel('log10(size) [phot]')
    cbar = fig.colorbar(pcm, ax=ax)
    ax.set_xlim([0, 500])
    ax.set_ylim([1.5, 4.5])
    cbar.set_label('std(E) / E')    

    fig, ax = plt.subplots(1,figsize=(10,8))
    pcm = ax.pcolormesh(yn, xn, n_energy, rasterized=True) #, vmin = 0, vmax=1) #, cmap='nipy_spectral')
    ax.set_xlabel('Impact parameter [m]')
    ax.set_ylabel('log10(size) [phot]')
    cbar = fig.colorbar(pcm, ax=ax)
    ax.set_xlim([0, 500])
    ax.set_ylim([1.5, 4.5])
    cbar.set_label('N') 


if __name__ == '__main__':

    parser = OptionParser()
    parser.add_option("-l", "--hillas", dest="hillas", help="path to a file with hillas parameters", default='../../../sst-1m_simulace/data_test/ryzen_testprod/0.0deg/Data/hillas_gamma_ze00_az000_p13_b07.npz')
    parser.add_option("-m", "--mc", dest="mc", help="path to a file with shower MC parameters", default='../../../sst-1m_simulace/data_test/ryzen_testprod/0.0deg/Data/shower_param_gamma_ze00_az000.txt')
    (options, args) = parser.parse_args()

    hillas = np.load(options.hillas) 
    mc = np.loadtxt(options.mc)
    
    # Reconstructed params
    size = hillas['size']
    border = hillas['border']

    # Masking borderflagged data
    #mask = [x == 0 or x == 1 for x in border]
    mask = [x == 0 for x in border]

    size = size[mask]
    mc = mc[mask,:]

    # True MC params
    core_distance = mc[:, 2]
    energy = mc[:, 3]
    x_core = mc[:, 9]
    y_core = mc[:, 10]
    theta = mc[:, 4]
    phi = mc[:, 5]


    # Impact parameter
    impact_parameter = impact_parameter(x_core, y_core, 0, 0, 4, theta, phi)    # not optimal, tel. coordinates should be loaded from somewhere..

    # Binning in log10 size
    size_bins_edges = np.linspace(0.5, 5, 100)

    # Binning in core distance
    impact_bins_edges = np.linspace(0, 500, 100)

    # Filling lookup tables [size, impact, value]
    binned_size_impact_energy = fill_lookup(size_bins_edges, impact_bins_edges, impact_parameter, np.log10(size), energy)

    # Save the lookup table
    suffix = 'ze00-az000-offset00.txt'
    path = '../../../sst-1m_simulace/data_test/ryzen_testprod/0.0deg/Data/'
    np.savetxt(path+'energy-lookup-gamma-'+suffix, binned_size_impact_energy, fmt='%.5f')

    # Plot
    
    # Look-up tables
    plot_lookup2d(binned_size_impact_energy)
    
    """
    fig = plt.figure(figsize=(10,8))
    plt.hist(np.log10(energy), histtype='step', stacked=True, fill=False, linewidth=4, color='black', bins=100)
    plt.yscale('log')
    
    fig = plt.figure(figsize=(10,8))
    plt.scatter(core_distance , np.log10(size), c=np.log10(energy), vmin = 0, vmax=np.log10(200))
    plt.xlabel('Core distance [m]')
    plt.ylabel('log10(size) [phot]')
    cbar = plt.colorbar()
    cbar.set_label("log10(E) [TeV]", labelpad=+1)

    fig = plt.figure(figsize=(10,8))
    plt.hist(core_distance)
    plt.xlabel('Core distance [m]')
    plt.ylabel('N')
    """
    
    plt.show()
