import numpy as np
from optparse import OptionParser
import matplotlib.pyplot as plt
import matplotlib as mpl

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
    
    # True MC params
    core_distance = mc[:, 2]
    energy = mc[:, 3]
    
    # Masking borderflagged data
    #mask = [x == 0 or x == 1 for x in border]
    mask = [x == 0 for x in border]
    
    
    # Plot
    fig = plt.figure(figsize=(10,8))
    plt.scatter(core_distance[mask] , np.log10(size[mask]), c=np.log10(energy[mask]), vmin = 0, vmax=np.log10(200))
    plt.xlabel('Core distance [m]')
    plt.ylabel('log10(size) [phot]')
    cbar = plt.colorbar()
    cbar.set_label("log10(E) [TeV]", labelpad=+1)

    
    fig = plt.figure(figsize=(10,8))
    plt.hist(core_distance[mask])
    plt.xlabel('Core distance [m]')
    plt.ylabel('N')
    
    plt.show()
