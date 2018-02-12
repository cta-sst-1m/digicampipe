import numpy as np
from optparse import OptionParser
import matplotlib.pyplot as plt
import matplotlib as mpl


def mask_ghosts(pix_x, pix_y, cen_x, cen_y, timing, max_dist):

    x_event = pix_x[timing > 0]
    y_event = pix_y[timing > 0]
    r = np.sqrt((x_event - cen_x)**2.0 + (y_event - cen_y)**2.0)
    mask = [x > max_dist for x in r]
    
    return mask


def time_rms(pix_x, pix_y, cen_x, cen_y, timing, max_dist):

    time_rms = []
    for i in range(len(timing[:,0])):

        # Get rid of residual ghost events
        mask = mask_ghosts(pix_x, pix_y, cen_x[i], cen_y[i], timing[i,:], max_dist)
        mask = np.array(mask)

        timing_event = timing[i, :]  * 4.0   # conversion of time 'slices' to ns
        timing_event = timing_event[timing_event > 0]
        timing_event = timing_event[~mask]

        if len(timing_event) > 10:  # selection of events detected in more than 10 pixels

            # Time RMS
            mean_time = np.mean(timing_event)
            sum_diff_time = np.dot((timing_event - mean_time),(timing_event - mean_time))
            time_rms.append(np.sqrt(sum_diff_time/len(timing_event))) 

    return time_rms



if __name__ == '__main__':

    parser = OptionParser()
    parser.add_option("-p", "--pixels", dest="pixels", help="path to a file with map of pixels", default='../../../sst-1m_simulace/data_test/ryzen_testprod/0.0deg/Data/pixels.txt')
    parser.add_option("-t", "--timp", dest="timing_prot", help="path to a file with timing of protons", default='../../../sst-1m_simulace/data_test/ryzen_testprod/0.0deg/Data/timing_proton_ze00_az000.txt')
    parser.add_option("-l", "--hilp", dest="hillas_prot", help="path to a file with hillas parameters of protons", default='../../../sst-1m_simulace/data_test/ryzen_testprod/0.0deg/Data//hillas_proton_ze00_az000_p13_b07.npz')
    parser.add_option("-g", "--timg", dest="timing_gamma", help="path to a file with timing of gamma", default='../../../sst-1m_simulace/data_test/ryzen_testprod/0.0deg/Data/timing_gamma_ze00_az000.txt')
    parser.add_option("-a", "--hilg", dest="hillas_gamma", help="path to a file with hillas parameters of gamma", default='../../../sst-1m_simulace/data_test/ryzen_testprod/0.0deg/Data//hillas_gamma_ze00_az000_p13_b07.npz')
    (options, args) = parser.parse_args()

    # pixel map
    pixels = np.loadtxt(options.pixels)
    pix_x = pixels[0, :]
    pix_y = pixels[1, :]

    hillas_prot = np.load(options.hillas_prot)
    timing_prot = np.loadtxt(options.timing_prot)
    hillas_gamma = np.load(options.hillas_gamma)
    timing_gamma = np.loadtxt(options.timing_gamma)

    size_prot = hillas_prot['size']
    border_prot = hillas_prot['border']
    size_gamma = hillas_gamma['size']
    border_gamma = hillas_gamma['border']
    cen_x_gamma = hillas_gamma['cen_x']
    cen_y_gamma = hillas_gamma['cen_y']
    cen_x_proton = hillas_prot['cen_x']
    cen_y_proton = hillas_prot['cen_y']

    min_size = 50

    # Masking border flagged events
    mask0_p = [x == 0 for x in border_prot]
    mask1_p = [x > min_size for x in size_prot]
    mask0_g = [x == 0 for x in border_gamma]
    mask1_g = [x > min_size for x in size_gamma]
    
    mask_p = ~np.isnan(cen_x_proton)*mask0_p*mask1_p
    mask_g = ~np.isnan(cen_x_gamma)*mask0_g*mask1_g

    timing_gamma = timing_gamma[mask_g,:]
    timing_prot = timing_prot[mask_p,:]
    cen_x_gamma = cen_x_gamma[mask_g]
    cen_y_gamma = cen_y_gamma[mask_g]
    cen_x_proton = cen_x_proton[mask_p]
    cen_y_proton =  cen_y_proton[mask_p]
    
    print('Gamma events ', len(timing_gamma))
    print('Proton events ', len(timing_prot))

    # RMS of individual events
    max_dist = 200  # mm, for getting rid of residual ghost events
    time_rms_gamma = time_rms(pix_x, pix_y, cen_x_gamma, cen_y_gamma, timing_gamma, max_dist)
    time_rms_proton = time_rms(pix_x, pix_y, cen_x_proton, cen_y_proton, timing_prot, max_dist)
    time_rms_gamma = np.array(time_rms_gamma)
    time_rms_proton = np.array(time_rms_proton)

    # Efficiency vs timeRMS cut
    # - ratio between N of events passing the cut and all events
    rms_cut = np.linspace(0, 20, 150)
    efficiency_gamma = []
    efficiency_proton = []
    for rc in rms_cut:
        efficiency_gamma.append(len(time_rms_gamma[time_rms_gamma < rc])/len(time_rms_gamma))
        efficiency_proton.append(len(time_rms_proton[time_rms_proton < rc])/len(time_rms_proton))
    efficiency_gamma = np.array(efficiency_gamma)
    efficiency_proton = np.array(efficiency_proton)

    # Quality factor
    quality = efficiency_gamma / np.sqrt(efficiency_proton)


    # PLOTS

    # Normalized histograms
    plt.figure(figsize=(11,8))
    weights_g = np.ones_like(time_rms_gamma)/float(len(time_rms_gamma))
    weights_p = np.ones_like(time_rms_proton)/float(len(time_rms_proton))
    plt.hist(time_rms_gamma, bins=50, alpha=0.5, weights=weights_g, label=str(len(timing_gamma))+' gamma', histtype='step', stacked=True, fill=False, linewidth=4, color='black')
    plt.hist(time_rms_proton, bins=50, alpha=0.5, weights=weights_p, label=str(len(timing_prot))+' proton', histtype='step', stacked=True, fill=False, linewidth=4, color='red')
    plt.xlim([0, 18])
    plt.legend()
    
    
    # Efficiency
    plt.figure(figsize=(11,8))
    plt.plot(rms_cut,efficiency_gamma, 'k.', label='gamma')
    plt.plot(rms_cut,efficiency_proton, 'r.', label='proton')
    plt.xlabel('TimeRMS cut [ns]')
    plt.ylabel('efficiency')
    plt.legend()
    
    # Quality factor
    plt.figure(figsize=(11,8))
    plt.plot(rms_cut,quality, 'k-')
    plt.xlabel('TimeRMS cut [ns]')
    plt.ylabel('Quality factor')

    plt.show()



