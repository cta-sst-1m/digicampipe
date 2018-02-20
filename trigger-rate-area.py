import numpy as np
from optparse import OptionParser
import matplotlib.pyplot as plt
import matplotlib as mpl
from shower_geometry import impact_parameter


def plot_energy_impact2d(data):  

    efficiency = data[:,4].reshape((len(np.unique(data[:,0])), len(np.unique(data[:,1]))))
    all_counts = data[:,2].reshape((len(np.unique(data[:,0])), len(np.unique(data[:,1]))))
    trigger_counts = data[:,3].reshape((len(np.unique(data[:,0])), len(np.unique(data[:,1]))))
    x, y = np.meshgrid(np.unique(data[:,1]), np.unique(data[:,0]))

    fig, ax = plt.subplots(1,figsize=(10,8))
    pcm = ax.pcolormesh(x, y, efficiency, rasterized=True) #, vmin = 0, vmax=1) #, cmap='nipy_spectral')
    ax.set_ylabel('Energy [TeV]')
    ax.set_xlabel('Impact parameter [m]')
    cbar = fig.colorbar(pcm, ax=ax)
    #ax.set_xlim([0, 500])
    #ax.set_ylim([1.5, 4.5])
    cbar.set_label('Efficiency')
    
    fig, ax = plt.subplots(1,figsize=(10,8))
    pcm = ax.pcolormesh(x, y, np.log10(all_counts), rasterized=True) #, vmin = 0, vmax=1) #, cmap='nipy_spectral')
    ax.set_ylabel('Energy [TeV]')
    ax.set_xlabel('Impact parameter [m]')
    cbar = fig.colorbar(pcm, ax=ax)
    #ax.set_xlim([0, 500])
    #ax.set_ylim([1.5, 4.5])
    cbar.set_label('log10 N_all')

    fig, ax = plt.subplots(1,figsize=(10,8))
    pcm = ax.pcolormesh(x, y, np.log10(trigger_counts), rasterized=True) #, vmin = 0, vmax=1) #, cmap='nipy_spectral')
    ax.set_ylabel('Energy [TeV]')
    ax.set_xlabel('Impact parameter [m]')
    cbar = fig.colorbar(pcm, ax=ax)
    #ax.set_xlim([0, 500])
    #ax.set_ylim([1.5, 4.5])
    cbar.set_label('log10 N_triggered')


# gamma spectrum (Crab at TeV energies)
def crab_spectrum(e_centers):

    dNdE_gamma = 2.83*10**-7 * (e_centers)**-2.62  #[1/(s m^2 TeV)]

    return dNdE_gamma


def binning_aeff(energy_all, energy_trig, impact_all, impact_trig, im_bins, im_centres, e_bins, e_centres):

    binned_efficiency = []
    A_eff = []

    for i in range(len(e_bins)-1):

        impact_all_bin = impact_all[(energy_all > e_bins[i]) & (energy_all <= e_bins[i+1])]
        impact_trig_bin = impact_trig[(energy_trig > e_bins[i]) & (energy_trig <= e_bins[i+1])]
        
        A = 0
        all_counts = []
        trig_counts = []
        P = []
        for j in range(len(im_bins)-1):

            all_counts.append(len(impact_all_bin[(impact_all_bin > im_bins[j]) & (impact_all_bin <= im_bins[j+1])]))
            trig_counts.append(len(impact_trig_bin[(impact_trig_bin > im_bins[j]) & (impact_trig_bin <= im_bins[j+1])]))

            if all_counts[j] == 0:
                all_counts[j] = np.nan
            P.append(trig_counts[j]/all_counts[j])
            binned_efficiency.append((e_centres[i], im_centres[j], all_counts[j], trig_counts[j], P[j]))
        P = np.array(P)
        
        if i == 0:
            P_all = P
        else: P_all = np.column_stack((P_all, P)) # trigger efficiency - impact parameter dependency for each energy bin

        # Effective area for each bin in impact, still energy dependent

        ind = np.argwhere(np.isnan(P))  # replacing nan values by linear interpolation of neighbours
        #print(len(P), len(im_bins)-1)
        for j in ind:
            if j > 0 and j < len(im_bins)-2:
                P[j] = (P[j-1] + P[j+1])/2.0
        r = im_centres
        dr = im_bins[1] - im_bins[0]
        A_eff.append(2 * np.pi * dr * sum(P * r))
        #suma = 0
        #for j in range(len(im_bins)-1):
        #    suma = suma + np.pi * P[j] * (im_bins[j+1]**2 - im_bins[j]**2)
        #A_eff.append(suma)

    binned_efficiency = np.array(binned_efficiency)
    P_all = np.array(P_all)

    return binned_efficiency, A_eff, P_all


if __name__ == '__main__':


    # !!! DONT FORGET TO SET UP RANGE OF HISTOGRAMS WITH REPSECT TO PRIMARY PARTICLE !!!

    parser = OptionParser()
    parser.add_option("-a", "--all", dest="all", help="path to a file with all MC events", default='../../../sst-1m_simulace/data_test/ryzen_testprod/0.0deg/Data/allmc_param_gamma_ze00_az000.txt')
    #parser.add_option("-d", "--digi", dest="trig_digi", help="path to a file with triggered events from digicampipe", default='../../../sst-1m_simulace/data_test/ryzen_testprod/0.0deg/Data/shower_param_gamma_ze00_az000.txt')
    (options, args) = parser.parse_args()

    # Load input data
    all_showers = np.loadtxt(options.all)
    triggered = all_showers[all_showers[:,9] == 1]

    energy_trig = triggered[:, 0]
    x_core_trig = triggered[:, 4]
    y_core_trig = triggered[:, 5]
    theta_trig = triggered[:,1]
    phi_trig = triggered[:,2]
    telpos_trig = triggered[:, 6:9]

    energy_all = all_showers[:,0]
    x_core_all = all_showers[:,4]
    y_core_all = all_showers[:,5]
    theta_all = all_showers[:,1]
    phi_all = all_showers[:,2]
    telpos_all = all_showers[:, 6:9]

    # Core distance
    core_dist_trig = np.sqrt((telpos_trig[:,0] - x_core_trig)**2 + (telpos_trig[:,1] - y_core_trig)**2)
    core_dist_all = np.sqrt((telpos_all[:,0] - x_core_all)**2 + (telpos_all[:,1] - y_core_all)**2)

    # Impact parameter
    impact_all = impact_parameter(x_core_all, y_core_all, telpos_all[:, 0], telpos_all[:, 1], telpos_all[:, 2], theta_all, phi_all)
    impact_trig = impact_parameter(x_core_trig, y_core_trig, telpos_trig[:, 0], telpos_trig[:, 1], telpos_trig[:, 2], theta_trig, phi_trig)

    # Binning in energy and impact parameter, effective area
    emin = 0.3  # TeV
    emax = 200
    imin = 0    # m
    imax = 1000
    im_bins = np.linspace(imin, imax, 30)                           # bins in impact parameters
    im_centres = (im_bins[:-1] + im_bins[1:])/2.0                   # centres of bins in impact
    e_log_bins = np.logspace(np.log10(emin), np.log10(emax), 30)    # LOGARITMIC bins in energy
    e_log_centres = (e_log_bins[:-1] + e_log_bins[1:])/2.0          # centres of bins in energy
    e_bins = np.linspace(emin, emax, 30)                            # linear bins in energy
    e_centres = (e_bins[:-1] + e_bins[1:])/2.0                      # centres

    binned_efficiency_log, A_eff_log, P_all_log = binning_aeff(energy_all, energy_trig, impact_all, impact_trig, im_bins, im_centres, e_log_bins, e_log_centres)
    binned_efficiency, A_eff, P_all = binning_aeff(energy_all, energy_trig, impact_all, impact_trig, im_bins, im_centres, e_bins, e_centres)


    print('ALL    TRIGG   Trig. eff')
    print(len(energy_all), len(energy_trig), len(energy_trig)/len(energy_all))  # Trigger efficiency

    # Logaritmic binning in energy
    emin = 0.3  # TeV
    emax = 200
    etl_counts, etl_edges = np.histogram(energy_trig,bins=np.logspace(np.log10(emin),np.log10(emax), 50))
    eal_counts, eal_edges = np.histogram(energy_all,bins=np.logspace(np.log10(emin),np.log10(emax), 50))
    etl_centres = (etl_edges[:-1] + etl_edges[1:])/2.0
    eal_centres = (eal_edges[:-1] + eal_edges[1:])/2.0
    efficiencyl = etl_counts/eal_counts

    # Linear binning in energy
    et_counts, et_edges = np.histogram(energy_trig,bins=np.linspace(emin, emax, 50))
    ea_counts, ea_edges = np.histogram(energy_all,bins=np.linspace(emin, emax, 50))
    et_centres = (et_edges[:-1] + et_edges[1:])/2.0
    ea_centres = (ea_edges[:-1] + ea_edges[1:])/2.0
    efficiency = et_counts/ea_counts
    # replacing nan values by linear interpolation of neighbours
    ind = np.argwhere(np.isnan(efficiency))
    efficiency[ind] = (efficiency[ind-1] + efficiency[ind+1])/2.0
 
    # Effective area - simple estimate according to D. Nedbal's disertation thesis
    a_eff_log = efficiencyl*(np.pi*1000**2)
    a_eff = efficiency*(np.pi*1000**2)
    
    # Linear binning of efficiency in impact parameter
    it_counts, it_edges = np.histogram(impact_trig,bins=np.linspace(imin, imax, 50))
    ia_counts, ia_edges = np.histogram(impact_all,bins=np.linspace(imin, imax, 50))
    it_centres = (it_edges[:-1] + it_edges[1:])/2.0
    efficiency_im = it_counts/ia_counts


    # Trigger rate - three diferent calculations
    # From simple estimate of a_eff, From integrated A_eff in linear and logaritmic binning

    dNdE_gamma = crab_spectrum(et_centres)
    dNdEs = a_eff*dNdE_gamma
    # integration over energy
    dE = et_edges[1] - et_edges[0]
    dNdt = sum(dNdEs*dE)
    print('Trigger rate [Hz]: ', dNdt)

    dNdE_gamma = crab_spectrum(e_centres)
    dNdEs = A_eff*dNdE_gamma
    # integration over energy
    dE = []
    for i in range(len(e_bins)-1):
        dE.append(e_bins[i+1] - e_bins[i])
    dNdt = sum(dNdEs*dE)
    print('Trigger rate [Hz]: ', dNdt)

    dNdE_gamma = crab_spectrum(e_log_centres)
    dNdEs = A_eff_log*dNdE_gamma
    # integration over energy
    dE = []
    for i in range(len(e_log_bins)-1):
        dE.append(e_log_bins[i+1] - e_log_bins[i])
    dNdt = sum(dNdEs*dE)
    print('Trigger rate [Hz]: ', dNdt)



    # PLOTS
    
    #2D histograms
    plot_energy_impact2d(binned_efficiency_log)

    # Effective area
    fig = plt.figure(figsize=(10,8))
    plt.plot(eal_centres, a_eff_log, 'r-', label='simple estimate')
    plt.plot(e_log_centres, A_eff_log, 'k-', label='integration')
    plt.ylabel('A eff [m^2]')
    plt.xlabel('E [TeV]')
    plt.xscale('log')
    plt.yscale('log')
    plt.legend()

    # Simulated vs triggered spectrum
    fig = plt.figure(figsize=(10,8))
    plt.plot(etl_centres, etl_counts, 'r.')
    plt.plot(eal_centres, eal_counts, 'k.')
    plt.ylabel('Events')
    plt.xlabel('E [TeV]')
    plt.xscale('log')
    plt.yscale('log')
    
    # Trigger efficiency - energy dependency
    fig = plt.figure(figsize=(10,8))
    plt.plot(etl_centres, efficiencyl, 'k.')
    plt.ylabel('trigger efficiency')
    plt.xlabel('E [TeV]')
    plt.xscale('log')
    plt.yscale('log')

    # Trigger efficiency - dependency on impact parameter
    fig = plt.figure(figsize=(10,8))
    plt.plot(it_centres, efficiency_im, 'k.')
    plt.ylabel('trigger efficiency')
    plt.xlabel('Impact parameter [m]')

    # Histogram of core distances
    fig = plt.figure(figsize=(10, 8))
    plt.hist(core_dist_trig, bins=100, histtype='step', stacked=True, fill=False, linewidth=4, color='red') #, range=[0, 500])
    plt.hist(core_dist_all, bins=100, histtype='step', stacked=True, fill=False, linewidth=4, color='black') #, range=[0, 500])
    plt.yscale('log')
    plt.xlabel('core distance [m]')
    plt.ylabel('N')

    fig = plt.figure(figsize=(10, 8))
    plt.hist(impact_trig, bins=100, histtype='step', stacked=True, fill=False, linewidth=4, color='red') #, range=[0, 500])
    plt.hist(impact_all, bins=100, histtype='step', stacked=True, fill=False, linewidth=4, color='black') #, range=[0, 500])
    plt.yscale('log')
    plt.xlabel('impact parameter [m]')
    plt.ylabel('N')

    # Trigger efficiency - impact parameter dependency in different energy bins
    # - useless for such a low number of triggered events on small impact
    """
    fig = plt.figure(figsize=(10, 8))
    for i in range(len(e_log_bins)-1):
        plt.plot(im_centres, P_all_log[:,i], '-')
    plt.xlabel('impact parameter [m]')
    plt.ylabel('triggerr eff')
    """

    # Histogram of impact parameter normalized by area
    bins = np.linspace(imin, imax, 30)
    N_norm_trig = []
    N_norm = []
    bins_mid = []
    for i in range(len(bins)-1):    
        bi_high = bins[i+1]
        bi_low = bins[i]
        area = np.pi * (bi_high**2 - bi_low**2)
        N_norm.append(len(impact_all[np.logical_and(impact_all >= bi_low, impact_all < bi_high)])/area)
        N_norm_trig.append(len(impact_trig[np.logical_and(impact_trig >= bi_low, impact_trig < bi_high)])/area)
        bins_mid.append((bins[i+1]-bins[i])/2.0 + bins[i])
    fig = plt.figure(figsize=(10, 8)) 
    plt.plot(bins_mid,N_norm, 'k.')
    plt.plot(bins_mid,N_norm_trig, 'r.')
    plt.yscale('log')
    #plt.ylim([0,1])  
    plt.xlabel('Impact all [m]')
    plt.ylabel('N/area')

    plt.show()
