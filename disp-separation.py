import numpy as np
from optparse import OptionParser
import matplotlib.pyplot as plt
import matplotlib as mpl


def plot_separation2d(data):  

    quality = data[:,4].reshape((len(np.unique(data[:,0])), len(np.unique(data[:,1]))))
    x, y = np.meshgrid(np.unique(data[:,1]), np.unique(data[:,0]))

    x_max = data[data[:,4] == max(data[:,4]), 1]
    y_max = data[data[:,4] == max(data[:,4]), 0]
    fig, ax = plt.subplots(1,figsize=(10,8))
    pcm = ax.pcolormesh(y, x, quality, rasterized=True) #, vmin = 0, vmax=1) #, cmap='nipy_spectral')
    ax.set_xlabel('slope')
    ax.set_ylabel('shift')
    plt.plot(y_max, x_max, 'r.')
    cbar = fig.colorbar(pcm, ax=ax)
    #ax.set_xlim([0, 500])
    #ax.set_ylim([1.5, 4.5])
    cbar.set_label('Quality factor')


if __name__ == '__main__':

    parser = OptionParser()
    parser.add_option("-l", "--hillasg", dest="hillas_gamma", help="path to a file with hillas parameters", default='../../../sst-1m_simulace/data_test/ryzen_testprod/0.0deg/Data/hillas_gamma_ze00_az000_p13_b07.npz')
    parser.add_option("-m", "--mcg", dest="mc_gamma", help="path to a file with shower MC parameters", default='../../../sst-1m_simulace/data_test/ryzen_testprod/0.0deg/Data/shower_param_gamma_ze00_az000.txt')
    parser.add_option("-a", "--hillasp", dest="hillas_proton", help="path to a file with hillas parameters", default='../../../sst-1m_simulace/data_test/ryzen_testprod/0.0deg/Data/hillas_proton_ze00_az000_p13_b07.npz')
    parser.add_option("-c", "--mcp", dest="mc_proton", help="path to a file with shower MC parameters", default='../../../sst-1m_simulace/data_test/ryzen_testprod/0.0deg/Data/shower_param_proton_ze00_az000.txt')
    (options, args) = parser.parse_args()
    
    hillas_gamma = np.load(options.hillas_gamma) 
    mc_gamma = np.loadtxt(options.mc_gamma)
    hillas_prot = np.load(options.hillas_proton) 
    mc_prot = np.loadtxt(options.mc_proton)


    size_prot = hillas_prot['size']
    border_prot = hillas_prot['border']
    size_gamma = hillas_gamma['size']
    border_gamma = hillas_gamma['border']
    cen_x_gamma = hillas_gamma['cen_x']
    cen_y_gamma = hillas_gamma['cen_y']
    cen_x_prot = hillas_prot['cen_x']
    cen_y_prot = hillas_prot['cen_y']
    cog_x_gamma = hillas_gamma['cen_x']  # in mm
    cog_y_gamma = hillas_gamma['cen_y']  #
    cog_x_prot = hillas_prot['cen_x']  # in mm
    cog_y_prot = hillas_prot['cen_y']  #
    length_gamma = hillas_gamma['length']
    width_gamma = hillas_gamma['width']
    length_prot = hillas_prot['length']
    width_prot = hillas_prot['width']

    min_size = 50

    # Masking border flagged events
    mask0_p = [x == 0 for x in border_prot]
    mask1_p = [x > min_size for x in size_prot]
    mask2_p = [~np.isnan(x)*~np.isinf(x) for x in length_prot/width_prot]
    mask0_g = [x == 0 for x in border_gamma]
    mask1_g = [x > min_size for x in size_gamma]
    mask2_g = [~np.isnan(x)*~np.isinf(x) for x in length_gamma/width_gamma]

    mask_p = ~np.isnan(cen_x_prot)*mask0_p*mask1_p*mask2_p
    mask_g = ~np.isnan(cen_x_gamma)*mask0_g*mask1_g*mask2_g

    cen_x_gamma = cen_x_gamma[mask_g]
    cen_y_gamma = cen_y_gamma[mask_g]
    cen_x_prot = cen_x_prot[mask_p]
    cen_y_prot =  cen_y_prot[mask_p]
    cog_x_gamma = cog_x_gamma[mask_g]  # in mm
    cog_y_gamma = cog_y_gamma[mask_g]  #
    cog_x_prot = cog_x_prot[mask_p]  # in mm
    cog_y_prot = cog_y_prot[mask_p]  #
    size_gamma = size_gamma[mask_g]
    width_gamma = width_gamma[mask_g]
    length_gamma = length_gamma[mask_g]
    size_prot = size_prot[mask_p]
    width_prot = width_prot[mask_p]
    length_prot = length_prot[mask_p]    
    
    print('Gamma events ', len(cen_x_gamma))
    print('Proton events ', len(cen_x_prot))

    # mc
    mc_prot = mc_prot[mask_p,:]
    mc_gamma = mc_gamma[mask_g,:]
    
    # True MC params
    x_offset_gamma= mc_gamma[:, 7]  # MC event source position, probably in deg
    y_offset_gamma = mc_gamma[:, 8]  #
    x_offset_prot= mc_prot[:, 7]  #
    y_offset_prot = mc_prot[:, 8]  #
    energy_gamma = mc_gamma[:, 3]
    energy_prot = mc_prot[:, 3]
    core_dist_gamma = mc_gamma[:, 2]
    core_dist_prot = mc_prot[:, 2]
    
    mm_to_deg = 0.24 / 24.3  # conversion of coordinates in mm to deg. Digicam: 0.24 deg/px, one SiPM is 24.3 mm wide
    cog_x_gamma = cog_x_gamma * mm_to_deg   # conversion to degrees
    cog_y_gamma = cog_y_gamma * mm_to_deg   # conversion to degrees
    cog_x_prot = cog_x_prot * mm_to_deg   # conversion to degrees
    cog_y_prot = cog_y_prot * mm_to_deg   # conversion to degrees

    disp_gamma = np.sqrt((x_offset_gamma - cog_x_gamma)**2.0 + (y_offset_gamma - cog_y_gamma)**2.0)     # apparently WRONG, it should be rewritten, according to propper handling with angular distances
    disp_prot = np.sqrt((x_offset_prot - cog_x_prot)**2.0 + (y_offset_prot - cog_y_prot)**2.0)          #

    """
    # OPTIMIZATION OF SEPARATION
    # separation by line: lw = A * disp + B
    B = np.linspace(-10, 3, 200)
    A = np.linspace(0.5, 10, 200)
    # mapping the parameter space
    parameter_space = []
    for i in range(len(A)):
        for j in range(len(B)):
            disp_g_sel = disp_gamma[length_gamma/width_gamma >= A[i] * disp_gamma + B[j]]
            disp_p_sel = disp_prot[length_prot/width_prot >= A[i] * disp_prot + B[j]]
            efficiency_g = len(disp_g_sel)/len(disp_gamma)
            efficiency_p = len(disp_p_sel)/len(disp_prot)
            quality_factor = efficiency_g/np.sqrt(efficiency_p)

            parameter_space.append((A[i], B[j], efficiency_g, efficiency_p, quality_factor))
    parameter_space = np.array(parameter_space)
    A_optim = parameter_space[parameter_space[:,4] == max(parameter_space[:,4]), 0]
    B_optim = parameter_space[parameter_space[:,4] == max(parameter_space[:,4]), 1]
    quality_optim = max(parameter_space[:,4])
    eff_g_optim = parameter_space[parameter_space[:,4] == max(parameter_space[:,4]), 2]
    eff_p_optim = parameter_space[parameter_space[:,4] == max(parameter_space[:,4]), 3]
    print(A_optim, B_optim, quality_optim, eff_g_optim, eff_p_optim)
    """
    A_optim = 3.69849246
    B_optim = -1.24623116


    # Saving positive/false detection flags according to Etienne's wish
    gamma_flag = []
    for i in range(len(disp_gamma)):
        if length_gamma[i]/width_gamma[i] > A_optim * disp_gamma[i] + B_optim:
            gamma_flag.append(1)
        else: gamma_flag.append(0)
    gamma_flag = np.array(gamma_flag)

    proton_flag = []
    for i in range(len(disp_prot)):
        if length_prot[i]/width_prot[i] < A_optim * disp_prot[i] + B_optim:
            proton_flag.append(1)
        else: proton_flag.append(0)
    proton_flag = np.array(proton_flag)

    print(len(gamma_flag[gamma_flag == 1]) / len(gamma_flag), len(proton_flag[proton_flag == 1]) / len(proton_flag))

    output_gamma = np.column_stack((energy_gamma, core_dist_gamma, gamma_flag))
    output_prot = np.column_stack((energy_prot, core_dist_prot, proton_flag))
    np.savetxt('../../../sst-1m_simulace/data_test/ryzen_testprod/0.0deg/Data/lw-disp-gamma.txt', output_gamma, fmt='%.5f %.5f %d')
    np.savetxt('../../../sst-1m_simulace/data_test/ryzen_testprod/0.0deg/Data/lw-disp-proton.txt', output_prot, fmt='%.5f %.5f %d')

    # PLOTS
    disp_smooth = np.linspace(0,4,10)

    #plot_separation2d(parameter_space)

    fig = plt.figure(figsize=(10,8))
    plt.hist2d(disp_gamma, length_gamma/width_gamma, bins=150, range=np.array([(0, 4), (0, 13)])) #, norm=mpl.colors.LogNorm()) #  range=np.array([(0, 4), (0, 0.3)]))
    plt.plot(disp_smooth, A_optim * disp_smooth + B_optim, 'w-')
    plt.colorbar()
    plt.xlim([0, 4])
    plt.xlabel('True DISP [deg]')
    plt.ylabel('length/width')

    fig = plt.figure(figsize=(10,8))
    plt.hist2d(disp_prot, length_prot/width_prot, bins=50, range=np.array([(0, 4), (0, 13)])) #, norm=mpl.colors.LogNorm()) #  range=np.array([(0, 4), (0, 0.3)]))
    plt.plot(disp_smooth, A_optim * disp_smooth + B_optim, 'w-')
    plt.colorbar()
    plt.xlim([0, 4])
    plt.xlabel('True DISP [deg]')
    plt.ylabel('length/width')

    plt.show()
