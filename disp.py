import numpy as np
from optparse import OptionParser
import matplotlib.pyplot as plt
import matplotlib as mpl

import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.optimize import minimize
from scipy.optimize import curve_fit
import scipy
from matplotlib.patches import Circle


def disp_eval(A, width, length, cog_x, cog_y, x_offset, y_offset, psi, skewness, size, method): 

    if method == 1:  # Simple method based on generaly known equation. DISP = f(width/length) for given THETA, AZ and offset
        disp_comp = A*(1 - width/length)
    elif method == 2:  # Based on eq. 2 in (Luke Riley St Marie, 2014). DISP = f(width/length, size) for given THETA, AZ and offset
        disp_comp = A[0] + np.log(size) * (A[1] + A[2]* (1 - width/(length + A[3]*np.log(size))))
    # Some tests
    elif method == 3:
        disp_comp = A[0] + A[1] * length/width
    elif method == 4:
        disp_comp = A[0] / (A[1] * width/length + A[2]) + A[3]
    elif method == 5:
        disp_comp = A[0] * np.exp(A[1] * ( 1 - width/length)) + A[2]
    #elif method == 6:
    #    disp_comp = A[0] + A[1] * length/width + A[2] * (length/width)**2 + A[3] * (length/width)**3

    x_source_comp0 = cog_x + disp_comp*np.cos(psi)  # Two possible solutions
    y_source_comp0 = cog_y + disp_comp*np.sin(psi)  #
    x_source_comp1 = cog_x - disp_comp*np.cos(psi)  #
    y_source_comp1 = cog_y - disp_comp*np.sin(psi)  #

    # Selection of one specific solution according to skewness
    # - head of the shower (close to the source) gives more signal
    # - if skewness > 0 head is closer to the center of FOV than tail
    x_source_comp = []
    y_source_comp = []
    for i in range(len(disp_comp)):
        if skewness[i] > 0:
            x_source_comp.append(x_source_comp1[i])
            y_source_comp.append(y_source_comp1[i])
        else:
            x_source_comp.append(x_source_comp0[i])
            y_source_comp.append(y_source_comp0[i])

    # Squared angular distance from the real source position
    theta_squared = (x_offset-x_source_comp)**2.0 + (y_offset-y_source_comp)**2.0

    # Optimalization criteria
    theta2_mean = np.mean(theta_squared)
    theta2_sum = sum(theta_squared)

    return disp_comp, theta_squared, x_source_comp, y_source_comp, theta2_sum



def disp_minimize(A, width, length, cog_x, cog_y, x_offset, y_offset, psi, skewness, size, method):

    disp_comp, theta_squared, x_source_comp, y_source_comp, theta2_sum = disp_eval(A, width, length, cog_x, cog_y, x_offset, y_offset, psi, skewness, size, method)

    return theta2_sum


def resolution(x,y):    # A simple way how to calculate "1-sigma" resolution
    
    center_x = np.mean(x)
    center_y = np.mean(y)
    x = x - center_x
    y = y - center_y
    N_full = len(x)

    r = 0.1
    N_in = 0

    while N_in < 0.68*N_full:

        N_in = len(x[(x**2 + y**2 < r**2)])
        r = r+0.005
        
    return r


# 2D Gaussian model
def res_gaussian(xy, x0, y0, sigma, H):

    x, y = xy
    theta_squared = (x0-x)**2.0 + (y0-y)**2.0

    I = H * np.exp(-theta_squared/(2.0*sigma**2.0))
    return I



if __name__ == '__main__':

    # A code for fitting of constants for the DISP method of image reconstruction. It can be fitted either on simulated data or real data of a point source after some reliable gamma/hadron separation.

    parser = OptionParser()
    parser.add_option("-l", "--hillas", dest="hillas", help="path to a file with hillas parameters", default='../../../sst-1m_simulace/data_test/ryzen_testprod/0.0deg/Data/hillas_gamma_ze00_az000_p13_b07.npz')
    parser.add_option("-m", "--mc", dest="mc", help="path to a file with shower MC parameters", default='../../../sst-1m_simulace/data_test/ryzen_testprod/0.0deg/Data/shower_param_gamma_ze00_az000.txt')
    parser.add_option("-e", "--eq", dest="method", help="Method/equation for the DISP parameter fit", default=1)
    
    (options, args) = parser.parse_args()
    
    hillas = np.load(options.hillas) 
    mc = np.loadtxt(options.mc)
    
    # Reconstructed params
    size = hillas['size']
    border = hillas['border']
    width = hillas['width']
    length = hillas['length']
    cog_x = hillas['cen_x']  #
    cog_y = hillas['cen_y']  # in mm
    psi = hillas['psi']
    skewness = hillas['skewness']  # represents assymetry of the image

    min_size = 50
    
    # Masking border flagged events
    mask0 = [x == 0 for x in border]
    mask1 = [x > 0.001 for x in width/length]  # because of some strange arteficial values..
    mask2 = [x > min_size for x in size]
    
    mask = ~np.isnan(width)*~np.isnan(cog_x)*mask0*mask1*mask2
    # hillas
    size = size[mask]
    width = width[mask]
    length = length[mask]
    psi = psi[mask]
    cog_x = cog_x[mask]  # in mm
    cog_y = cog_y[mask]  #
    skewness = skewness[mask]
    # mc
    mc = mc[mask,:]
    
    # True MC params
    core_distance = mc[:, 2]
    energy = mc[:, 3]
    x_offset = mc[:, 7]  # MC event source position, probably in deg
    y_offset = mc[:, 8]  #
    thetap = mc[:, 4]
    phi = mc[:, 5]
    core_distance = mc[:, 2]
    
    mm_to_deg = 0.24 / 24.3  # conversion of coordinates in mm to deg. Digicam: 0.24 deg/px, one SiPM is 24.3 mm wide
    cog_x = cog_x * mm_to_deg   # conversion to degrees
    cog_y = cog_y * mm_to_deg   # conversion to degrees

    disp = np.sqrt((x_offset - cog_x)**2.0 + (y_offset - cog_y)**2.0)

    
    # Minimization of disp
    if options.method == 1:

        res = minimize_scalar(lambda A: disp_minimize(A, width, length, cog_x, cog_y, x_offset, y_offset, psi, skewness, size, options.method))
        disp_comp, theta_squared, x_source_comp, y_source_comp, theta2_sum = disp_eval(res.x, width, length, cog_x, cog_y, x_offset, y_offset, psi, skewness, size, options.method)

    elif options.method == 2 or options.method == 4:

        x0 = [1,1,1,1]
        res = minimize(disp_minimize, x0, args=(width, length, cog_x, cog_y, x_offset, y_offset, psi, skewness, size, options.method))
        disp_comp, theta_squared, x_source_comp, y_source_comp, theta2_sum = disp_eval(res.x, width, length, cog_x, cog_y, x_offset, y_offset, psi, skewness, size, options.method)

    elif options.method == 3:

        x0 = [1,1]
        res = minimize(disp_minimize, x0, args=(width, length, cog_x, cog_y, x_offset, y_offset, psi, skewness, size, options.method))
        disp_comp, theta_squared, x_source_comp, y_source_comp, theta2_sum = disp_eval(res.x, width, length, cog_x, cog_y, x_offset, y_offset, psi, skewness, size, options.method)
    
    elif options.method == 5:
        x0 = [1,1,1]
        res = minimize(disp_minimize, x0, args=(width, length, cog_x, cog_y, x_offset, y_offset, psi, skewness, size, options.method))
        disp_comp, theta_squared, x_source_comp, y_source_comp, theta2_sum = disp_eval(res.x, width, length, cog_x, cog_y, x_offset, y_offset, psi, skewness, size, options.method)


    # RESOLUTION
    
    # Radius of a circle containing 68% of the signal
    resolution = resolution(x_source_comp, y_source_comp)
    print('R[deg] contains 68% of all events: ', resolution)

    # 2D Gaussian fit 
    # - creation of a matrix
    bins = 150
    xy_min = -1.0
    xy_max = 1.0
    n_bin = np.histogram2d(x_source_comp, y_source_comp, bins=bins, range=[[xy_min, xy_max], [xy_min, xy_max]])
    n_bin_values = n_bin[0]

    # - coordinates of middle of each interval
    x = n_bin[1][:-1] + (n_bin[1][1]-n_bin[1][0])/2.0
    y = n_bin[2][:-1] + (n_bin[2][1]-n_bin[2][0])/2.0
    xx, yy = np.meshgrid(x, y)
    
    initial_guess = [0, 0, 0.1, 10] # [x0, y0, sigma, amplitude]
    gauss_params, uncert_cov = curve_fit(res_gaussian, (xx.ravel(), yy.ravel()), n_bin_values.ravel(), p0=initial_guess)
    print('Sigma[deg] from a fit with 2D gaussian: ', gauss_params[2])
    

    # PLOTS

    fig = plt.figure(figsize=(10,8))
    plt.hist2d(disp, length/width, bins=150, range=np.array([(0, 4), (0, 13)])) #, norm=mpl.colors.LogNorm()) #  range=np.array([(0, 4), (0, 0.3)]))
    plt.colorbar()
    plt.xlabel('True DISP [deg]')
    plt.ylabel('length/width')

    fig = plt.figure(figsize=(10,8))
    plt.hist2d(disp, width/length, bins=150) #, norm=mpl.colors.LogNorm()) #  range=np.array([(0, 4), (0, 0.3)]))
    plt.colorbar()
    plt.xlabel('True DISP [deg]')
    plt.ylabel('width/length')

    fig = plt.figure(figsize=(10,8))
    plt.hist2d(disp, 1 - width/length, bins=150) #, norm=mpl.colors.LogNorm()) #  range=np.array([(0, 4), (0, 0.3)]))
    plt.colorbar()
    plt.xlabel('True DISP [deg]')
    plt.ylabel('1 - width/length')

    fig = plt.figure(figsize=(10,8))
    plt.hist2d(np.log(size), width/length, bins=150) #, norm=mpl.colors.LogNorm()) #  range=np.array([(0, 4), (0, 0.3)]))
    plt.colorbar()
    plt.xlabel('log size')
    plt.ylabel(' width/length')

    fig = plt.figure(figsize=(10,8))
    plt.hist2d(np.log(size), disp, bins=150) #, norm=mpl.colors.LogNorm()) #  range=np.array([(0, 4), (0, 0.3)]))
    plt.colorbar()
    plt.xlabel('log size')
    plt.ylabel('True DISP')


    # Statistics for all methods

    fig, ax = plt.subplots(1, 3,figsize=(30,7))
    #fig.suptitle("disp_comp = A*(1 - width/length)", fontsize=18)

    ax[0].hist(theta_squared,bins=50, alpha=0.5, log=True)
    ax[0].set_xlim([0, 20])
    ax[0].set_ylim(ymin=1)
    ax[0].set_xlabel('theta^2 [deg^2]')

    ax[1].hist2d(x_source_comp, y_source_comp, bins=150, range=np.array([(-1, 1), (-1, 1)]))
    circle = Circle((gauss_params[0], gauss_params[1]), gauss_params[2], facecolor='none', edgecolor="red", linewidth=1, alpha=0.8)
    ax[1].add_patch(circle)
    ax[1].scatter(gauss_params[0], gauss_params[1], s=30, c="red", marker="+", linewidth=1)
    ax[1].set_xlabel('x [deg]')
    ax[1].set_ylabel('y [deg]')

    ax[2].hist2d(disp, disp_comp, bins=150, range=np.array([(0, 3), (0, 3)]))
    ax[2].plot([0,3],[0,3],'r-')
    ax[2].set_xlabel('True DISP [deg]')
    ax[2].set_ylabel('Comp DISP [deg]')


    """    
    for event_no in range(21,30):
        plot_event(pix_x, pix_y, image, event_no)
        plot_hillas(hillas, event_no)
        ax1 = plt.gca()
        ax1.plot(x_source_comp[event_no],y_source_comp[event_no],'r.')
        #ax1.plot(x_source_comp1[event_no],y_source_comp1[event_no],'g.')
        #print(x_source_comp[event_no],y_source_comp[event_no])
    """
    
    plt.show()
