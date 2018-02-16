import numpy as np
from optparse import OptionParser
import matplotlib.pyplot as plt
import matplotlib as mpl

import pandas as pd


if __name__ == '__main__':

    parser = OptionParser()
    parser.add_option("-l", "--hillas", dest="hillas", help="path to a file with hillas parameters", default='../../../sst-1m_simulace/data_test/ryzen_testprod/0.0deg/Data/hillas_proton_ze00_az000_p13_b07.npz')
    parser.add_option("-m", "--mc", dest="mc", help="path to a file with shower MC parameters", default='../../../sst-1m_simulace/data_test/ryzen_testprod/0.0deg/Data/shower_param_proton_ze00_az000.txt')
    (options, args) = parser.parse_args()
    
    hillas = np.load(options.hillas) 
    mc = np.loadtxt(options.mc)
    
    # Reconstructed params
    size = hillas['size']
    border = hillas['border']
    width = hillas['width']
    length = hillas['length']
    cog_x = hillas['cen_x']  # in mm
    cog_y = hillas['cen_y']  #
    skewness = hillas['skewness']  # represents assymetry of the image
    
    # True MC params
    core_distance = mc[:, 2]
    energy = mc[:, 3]
    x_offset = mc[:, 7]  # MC event source position, probably in deg
    y_offset = mc[:, 8]  #
    thetap = mc[:, 4]
    phi = mc[:, 5]
    core_distance = mc[:, 2]

    mm_to_deg = 0.24 / 24.3  # conversion of coordinates in mm to deg. Digicam: 0.24 deg/px, one SiPM is 24.3 mm wide
    
    disp = np.sqrt((x_offset - cog_x)**2.0 + (y_offset - cog_y)**2.0) * mm_to_deg  # conversion to degrees included
    
    min_size = 50
    
    # Masking border flagged events
    mask0 = [x == 0 for x in border]
    mask1 = [x > 0.001 for x in width/length]  # because of some strange arteficial values..
    mask2 = [x > min_size for x in size]
    
    mask = ~np.isnan(width)*~np.isnan(disp) #*mask0*mask1*mask2
    size = size[mask]
    disp = disp[mask]
    width = width[mask]
    length = length[mask]
    thetap = thetap[mask]
    phi = phi[mask]
    core_distance = core_distance[mask]
    skewness = skewness[mask]
    energy = energy[mask]
    
    """
    # Events behind the border even after the bordercut
    energy_strange = energy[thetap > 5]  # double identification
    width_strange = width[thetap > 5]      #
    n = 0
    for i in range(len(energy_strange)):
        if list(mc[:, 3]).index(energy_strange[i]) == list(hillas['width']).index(width_strange[i]):
            print(list(mc[:, 3]).index(energy_strange[i])) # print N of lines with the events
            n+=1
    """
    
    """
    # Binning in size
    # - divide data into unequal sized bins with the same number of datapoint
    
    # parameter q specifies the number of bins
    out, bin_edges = pd.qcut(size, q=20, precision=1, retbins=True)
    
    for i in range(len(bin_edges)-1):
        edge_min = bin_edges[i]
        edge_max = bin_edges[i+1]
        w_bin = width[np.logical_and(size >= edge_min, size < edge_max)]
        l_bin = length[np.logical_and(size >= edge_min, size < edge_max)]
        disp_bin = disp[np.logical_and(size >= edge_min, size < edge_max)]
        
        fig = plt.figure(figsize=(10,8))
        plt.hist2d(disp_bin, l_bin/w_bin, bins=30, range=np.array([(0, 4), (0, 14)]))  
        plt.text(2.8, 12.9, 'size bin', fontsize=30, color='white', weight='bold')      
        plt.text(2.8, 11.83, str(int(edge_min)), fontsize=30, color='white', weight='bold')
        plt.text(2.8, 10.75, str(int(edge_max)), fontsize=30, color='white', weight='bold')
        plt.colorbar()
        plt.xlabel('DISP [deg]')
        plt.ylabel('length/width')
        plt.savefig('../../../sst-1m_simulace/data_test/ryzen_testprod/0.0deg/Data/lw-disp-size' + str(int(edge_min)) + '-' + str(int(edge_max)) + '.png')
    """
    

    # Plots
    
    fig = plt.figure(figsize=(10, 8))
    #plt.hist2d(disp, length/width, bins=70) #, norm=mpl.colors.LogNorm()) #  range=np.array([(0, 4), (0, 0.3)]))
    plt.hist2d(disp, length/width, bins=70, range=np.array([(0, 4), (0, 13)]))
    #plt.plot(disp, width/length,'.')
    plt.colorbar()
    plt.xlabel('DISP [deg]')
    plt.ylabel('length/width')
    
    
    fig = plt.figure(figsize=(10,8))
    plt.hist(size, bins=20)
    plt.xlabel('size')
    plt.ylabel('N')
    
    # Histogram normalized by area
    # - this can be used for measurement of the acceptance
    bins = range(0, 11)
    bins_mid = np.linspace(min(bins)+0.5, max(bins)+0.5, len(bins))
    N = []
    N_norm = []
    for i in range(len(bins)-1):    
        bi_high = bins[i+1]
        bi_low = bins[i]
        area = np.pi * (bi_high**2 - bi_low**2)
        N.append(len(thetap[np.logical_and(thetap >= bi_low, thetap < bi_high)]))
        N_norm.append(len(thetap[np.logical_and(thetap >= bi_low, thetap < bi_high)])/area)
    
    fig = plt.figure(figsize=(10, 8)) 
    plt.plot(bins_mid[:-1],N_norm)    
    plt.xlabel('thetap')
    plt.ylabel('N/area')
    
    
    # Histogram of zenit angles
    fig = plt.figure(figsize=(10, 8))
    plt.hist(thetap, bins=50)
    plt.xlabel('thetap')
    plt.ylabel('N')
    
    # Histogram of core distances
    fig = plt.figure(figsize=(10, 8))
    plt.hist(core_distance, bins=100, histtype='step', stacked=True, fill=False, linewidth=4, color='black') #, range=[0, 500])
    plt.yscale('log')
    plt.xlabel('core distance [m]')
    plt.ylabel('N')
    
    # Disp - core distance dependence
    fig = plt.figure(figsize=(10, 8))
    #plt.hist2d(disp, length/width, bins=70) #, norm=mpl.colors.LogNorm()) #  range=np.array([(0, 4), (0, 0.3)]))
    plt.hist2d(disp, core_distance, bins=70, range=np.array([(0, 4), (0, 400)]))
    #plt.plot(disp, width/length,'.')
    plt.colorbar()
    plt.xlabel('DISP [deg]')
    plt.ylabel('core distance [m]')
    
    
    fig = plt.figure(figsize=(10, 8))
    #plt.hist2d(disp, length/width, bins=70) #, norm=mpl.colors.LogNorm()) #  range=np.array([(0, 4), (0, 0.3)]))
    plt.hist2d(disp, skewness, bins=30) #, range=np.array([(0, 4), (0, 400)]))
    #plt.plot(disp, width/length,'.')
    plt.colorbar()
    plt.xlabel('DISP [deg]')
    plt.ylabel('skewness')
    
    
    plt.show()
