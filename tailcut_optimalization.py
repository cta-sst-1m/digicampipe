import numpy as np
from optparse import OptionParser
import matplotlib.pyplot as plt
import matplotlib as mpl
import os

def plot_rms2d(data):  

    rms2 = data[:,2].reshape((len(np.unique(data[:,0])), len(np.unique(data[:,1]))))
    x, y = np.meshgrid(np.unique(data[:,1]), np.unique(data[:,0]))

    fig = plt.figure(figsize=(9, 8))
    ax1 = fig.add_subplot(111)
    pcm = ax1.pcolormesh(x, y, np.log10(rms2), rasterized=True) #, vmin = 0, vmax=1) #, cmap='nipy_spectral')
    ax1.set_ylabel('high level')
    ax1.set_xlabel('low level')
    ax1.invert_yaxis()
    cbar = fig.colorbar(pcm)
    cbar.set_label('log10 RMS')

def plot_len2d(data):  

    rms2 = data[:,2].reshape((len(np.unique(data[:,0])), len(np.unique(data[:,1]))))
    x, y = np.meshgrid(np.unique(data[:,1]), np.unique(data[:,0]))

    fig = plt.figure(figsize=(9, 8))
    ax1 = fig.add_subplot(111)
    pcm = ax1.pcolormesh(x, y, rms2, rasterized=True) #, vmin = 0, vmax=1) #, cmap='nipy_spectral')
    ax1.set_ylabel('high level')
    ax1.set_xlabel('low level')
    ax1.invert_yaxis()
    cbar = fig.colorbar(pcm)
    cbar.set_label('N events')


if __name__ == '__main__':

    parser = OptionParser()
    parser.add_option("-l", "--hillas", dest="hillas", help="path to a files with hillas parameters", default='../../../sst-1m_simulace/results/tailcut_optimalization/')
    (options, args) = parser.parse_args()
    
    all_file_list = os.listdir(options.hillas)
    all_file_list =  sorted(all_file_list)  # Neccessary for correct mesh plotting !
        
    rms = []
    high = []
    low = []
    mean_miss = []
    len_miss = []

    for fi in all_file_list:
        #print(fi)
        hillas = np.load(options.hillas + fi) 
        miss = hillas['miss'] / 24.3 * 0.24  # conversion to degrees included
        
        rms.append(np.sqrt(np.dot(miss,miss)/len(miss)))
        mean_miss.append(np.mean(miss))
        len_miss.append(len(miss))
        high.append(int(fi[22:24]))
        low.append(int(fi[26:28]))

    rms = np.array(rms)
    mean_miss = np.array(mean_miss)
    len_miss = np.array(len_miss)
    low = np.array(low)
    high = np.array(high)

    data = np.array(np.column_stack((high.T,low.T,rms.T)))
    data2 = np.array(np.column_stack((high.T,low.T,mean_miss.T)))
    data3 = np.array(np.column_stack((high.T,low.T,len_miss.T)))

    print(data[rms == min(rms),:])

    """
    fig = plt.figure(figsize=(10,8))
    plt.scatter(low , high, c=rms) #, vmin = 0, vmax=1)
    plt.xlabel('low level')
    plt.ylabel('high level')
    cbar = plt.colorbar()
    cbar.set_label("RMS", labelpad=+1)
    """
    fig = plt.figure(figsize=(10,8))
    plt.hist(miss, bins=50, histtype='step', stacked=True, fill=False, linewidth=4, color='black', range=[0, 1.5])
    plt.xlabel('miss [deg]')
    plt.ylabel('N')

    plot_rms2d(data)
    plot_len2d(data3)

    plt.show()
