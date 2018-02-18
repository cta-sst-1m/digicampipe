import numpy as np
import events_image
from optparse import OptionParser
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.stats import binned_statistic
from shower_geometry import impact_parameter


def rotate_around_point(point, radians, origin=(0, 0)):

    x, y = point
    ox, oy = origin

    qx = ox + np.cos(radians) * (x - ox) + np.sin(radians) * (y - oy)
    qy = oy + -np.sin(radians) * (x - ox) + np.cos(radians) * (y - oy)

    return qx, qy


def alpha_etienne2(psi, cen_x, cen_y, source_x, source_y):  # etienne's code from scan_crab_cluster.c

    d_x = np.cos(psi)
    d_y = np.sin(psi)
    to_c_x = source_x - cen_x
    to_c_y = source_y - cen_y
    to_c_norm = np.sqrt(to_c_x**2.0 + to_c_y**2.0)
    to_c_x = to_c_x/to_c_norm
    to_c_y = to_c_y/to_c_norm
    p_scal_1 = d_x*to_c_x + d_y*to_c_y
    p_scal_2 = -d_x*to_c_x + -d_y*to_c_y
    alpha_c_1 = abs(np.arccos(p_scal_1))
    alpha_c_2 = abs(np.arccos(p_scal_2))
    alpha_cetienne = alpha_c_1
    for i in range(len(alpha_cetienne)):
        if (alpha_c_2[i] < alpha_c_1[i]):
            alpha_cetienne[i] = alpha_c_2[i]
    alpha = 180.0/np.pi*alpha_cetienne

    return alpha


if __name__ == '__main__':

    parser = OptionParser()
    parser.add_option("-p", "--pixels", dest="pixels", help="path to a file with map of pixels", default='../../../sst-1m_simulace/data_test/ryzen_testprod/0.0deg/Data/pixels.txt')
    parser.add_option("-l", "--hillas", dest="hillas", help="path to a file with hillas parameters", default='../../../sst-1m_simulace/data_test/ryzen_testprod/0.0deg/Data//hillas_gamma_ze00_az000_p13_b07.npz')
    parser.add_option("-t", "--timing", dest="timing", help="path to a file with timing", default='../../../sst-1m_simulace/data_test/ryzen_testprod/0.0deg/Data/timing_gamma_ze00_az000.txt')
    parser.add_option("-m", "--mc", dest="mc", help="path to a file with shower MC parameters", default='../../../sst-1m_simulace/data_test/ryzen_testprod/0.0deg/Data/shower_param_gamma_ze00_az000.txt')
    (options, args) = parser.parse_args()
   
    pixels = np.loadtxt(options.pixels)
    hillas = np.load(options.hillas)   
    timing = np.loadtxt(options.timing)
    mc = np.loadtxt(options.mc)
    
    # pixel map
    pix_x = pixels[0, :]
    pix_y = pixels[1, :]
    
    # Hillas parameters
    psi = hillas['psi']
    cen_x = hillas['cen_x']
    cen_y = hillas['cen_y']
    size = hillas['size']
    border = hillas['border']

    min_size = 50

    # Masking border flagged events
    mask0 = [x == 0 for x in border]
    mask2 = [x > min_size for x in size]

    mask = ~np.isnan(cen_x)*mask0*mask2

    psi = psi[mask]
    cen_x = cen_x[mask]
    cen_y = cen_y[mask]
    size = size[mask]
    mc = mc[mask,:]
    timing = timing[mask,:]

    # MC parameters
    x_offset = mc[:, 7]  # MC event source position
    y_offset = mc[:, 8]  #
    x_core = mc[:, 9]
    y_core = mc[:, 10]
    theta = mc[:, 4]
    phi = mc[:, 5]

    alpha = alpha_etienne2(psi, cen_x, cen_y, x_offset, y_offset)
    impact_parameter = impact_parameter(x_core, y_core, 0, 0, 4, theta, phi)    # not optimal, tel. coordinates should be loaded from somewhere..

    mm_to_deg = 0.24 / 24.3  # conversion of coordinates in mm to deg. Digicam: 0.24 deg/px, one SiPM is 24.3 mm wide

    time_gradient = []
    impact_parameter_sel = []
    disp = []

    for i in range(len(timing[:,0])):

        # Get rid of residual ghost events
        max_dist = 200  # mm
        x_event = pix_x[timing[i, :] > 0]
        y_event = pix_y[timing[i, :] > 0]
        r = np.sqrt((x_event - cen_x[i])**2.0 + (y_event - cen_y[i])**2.0)
        mask = [x > max_dist for x in r]
        mask = np.array(mask)

        if psi[i] != 0:
            
            # Center of rotation
            # - as crossection of main axis of the Hillas ellipse and y = 0 axis
            y_rot_cen = 0
            x_rot_cen = cen_x[i] - cen_y[i]/np.tan(psi[i])

            # Rotation of event with PSI angle
            # - after this operation the main axis of all events lies in y=0 coordinates
            x_rot, y_rot = rotate_around_point((pix_x, pix_y), psi[i], origin=(x_rot_cen, y_rot_cen))

        else: 
            x_rot = pix_x
            y_rot = pix_y

        timing_event = timing[i, :]  * 4.0   # conversion of time 'slices' to ns
        x_rot_sel = x_rot[timing[i, :] > 0] * mm_to_deg  # conversion of coordinates in mm to deg
        y_rot_sel = y_rot[timing[i, :] > 0] * mm_to_deg
        timing_event = timing_event[timing_event > 0]

        if len(timing_event > 0):   # Because sometimes the timing_event matrix is empty. Timing matrix is full of zeros because of failure of Hillas parameter fit in simtel_pipeline

            if len(x_rot_sel[~mask]) > 10 and alpha[i] < 90:  # selection cuts
                fit = np.polyfit(x_rot_sel[~mask], timing_event[~mask], 1)
                time_gradient.append(fit[0])
                impact_parameter_sel.append(impact_parameter[i])

                # Disp
                disp.append(np.sqrt((x_offset[i] - cen_x[i])**2.0 + (y_offset[i] - cen_y[i])**2.0) * mm_to_deg)  # conversion to degrees included


    print(len(time_gradient))

    
    # Binning in Impact parameter
    bin_time = binned_statistic(impact_parameter_sel, time_gradient, bins=20, range=(0, 400))
    bin_impact = binned_statistic(impact_parameter_sel, impact_parameter_sel, bins=20, range=(0, 400))[0]
    bin_means = bin_time[0]
    bin_edges = bin_time[1]

    bin_means_std = []
    time_gradient = np.array(time_gradient)
    for i in range(len(bin_edges)-1):
        edge_min = bin_edges[i]
        edge_max = bin_edges[i+1]
        bin_means_std.append(np.std(time_gradient[np.logical_and(impact_parameter_sel >= edge_min, impact_parameter_sel < edge_max)]))
    bin_means_std = np.array(bin_means_std)

    """
    # Faster way how to compute means and stds of binned data
    n, _ = np.histogram(impact_parameter_sel, bins=20)
    sy, _ = np.histogram(impact_parameter_sel, bins=20, weights=time_gradient)
    sy2, _ = np.histogram(impact_parameter_sel, bins=20, weights=time_gradient*time_gradient)
    mean = sy / n
    std = np.sqrt(sy2/n - mean*mean)
    """
    
    # Approximation of the Time gradient -impact parameter dependence
    fit = np.polyfit(bin_impact[~np.isnan(bin_impact)], bin_means[~np.isnan(bin_impact)], 5)
    # Approximation of the sigma - impact parameter dependence
    fit_sigma = np.polyfit(bin_impact[~np.isnan(bin_impact)], bin_means_std[~np.isnan(bin_impact)], 2)


    # Scaled TimeGradient
    scaled_timegrad = (time_gradient - np.polyval(fit, impact_parameter_sel)) / np.polyval(fit_sigma, impact_parameter_sel)



    # PLOTS

    # Time gradient hist
    plt.figure(figsize=(11,8))
    plt.hist2d(impact_parameter_sel, time_gradient, bins=150, range=np.array([(0, 400), (-40, 40)])) #, norm=mpl.colors.LogNorm())
    plt.colorbar()
    plt.xlabel('impact_parameter [m]')
    plt.ylabel('time gradient [ns/deg]')

    # x,y rot coordinates distribution
    plt.figure(figsize=(11,8))
    plt.hist(y_rot_sel, bins=10, histtype='step', stacked=True, fill=False, linewidth=4, color='black')
    #plt.xlim([-2, 2])
    plt.xlabel('y_rot_sel [deg]')

    plt.figure(figsize=(11,8))
    plt.hist(x_rot_sel, bins=10, histtype='step', stacked=True, fill=False, linewidth=4, color='black')
    #plt.xlim([-2, 2])
    plt.xlabel('x_rot_sel [deg]')    
    

    # Time gradient plot
    plt.figure(figsize=(11,8))
    plt.errorbar(bin_impact, bin_means, yerr=bin_means_std, fmt='.', ms=15)
    plt.plot(bin_impact[~np.isnan(bin_impact)], np.polyval(fit, bin_impact[~np.isnan(bin_impact)]),'r-')
    #plt.errorbar((_[1:] + _[:-1])/2, mean, yerr=std, fmt='r-')
    plt.xlabel('impact_parameter [m]')
    plt.ylabel('time gradient [ns/deg]')

    # Sigma - impact parameter dependency
    plt.figure(figsize=(11,8))
    plt.plot(bin_impact, bin_means_std, 'k.')
    plt.plot(bin_impact[~np.isnan(bin_impact)], np.polyval(fit_sigma, bin_impact[~np.isnan(bin_impact)]),'k-')
    plt.xlabel('impact_parameter [m]')
    plt.ylabel('sigma [ns/deg]')

    # Scaled TimeGradient hist
    plt.figure(figsize=(11,8))
    weights = np.ones_like(scaled_timegrad)/float(len(scaled_timegrad))
    plt.hist(scaled_timegrad, bins=50, weights=weights, histtype='step', stacked=True, fill=False, linewidth=4, color='black')
    #plt.xlim([0, 18])
    plt.xlabel('(TG - <TG>) / sigma')
    #plt.legend()


    # DISP vs Time gradient
    plt.figure(figsize=(11, 8))
    plt.hist2d(disp, time_gradient, bins=50, range=np.array([(0, 4), (-40, 40)])) #, norm=mpl.colors.LogNorm())
    plt.colorbar()
    plt.xlabel('DISP [deg]')
    plt.ylabel('time gradient [ns/deg]')

    # Binning in DISP
    bin_time = binned_statistic(disp, time_gradient, bins=20, range=(0, 4))
    bin_disp = binned_statistic(disp, disp, bins=20, range=(0, 4))[0]
    bin_means = bin_time[0]
    bin_edges = bin_time[1]
    
    bin_means_std = []
    for i in range(len(bin_edges)-1):
        edge_min = bin_edges[i]
        edge_max = bin_edges[i+1]
        bin_means_std.append(np.std(time_gradient[np.logical_and(disp >= edge_min, disp < edge_max)]))

    # Time gradient DISP plot
    plt.figure(figsize=(11,8))
    plt.errorbar(bin_disp, bin_means, yerr=bin_means_std, fmt='.', ms=15)
    #plt.plot(bin_impact[~np.isnan(bin_impact)], np.polyval(fit, bin_impact[~np.isnan(bin_impact)]),'r-')
    #plt.errorbar((_[1:] + _[:-1])/2, mean, yerr=std, fmt='r-')
    plt.xlabel('DISP [deg]')
    plt.ylabel('time gradient [ns/deg]')

    plt.show()
    
    
    
    
    
