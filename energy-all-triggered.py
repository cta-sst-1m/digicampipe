import numpy as np
from optparse import OptionParser
import matplotlib.pyplot as plt
import matplotlib as mpl

if __name__ == '__main__':


    # !!! DONT FORGET TO SET UP RANGE OF HISTOGRAMS WITH REPSECT TO PRIMARY PARTICLE !!!

    parser = OptionParser()
    parser.add_option("-a", "--all", dest="all", help="path to a file with all MC events", default='../../../sst-1m_simulace/data_test/ryzen_testprod/0.0deg/Data/shower_allmc_param_gamma_ze00_az000.txt')
    (options, args) = parser.parse_args()
    
    all_showers = np.loadtxt(options.all)
    triggered = all_showers[all_showers[:,6] == 1]
    
    energy_trig = triggered[:, 0]
    energy_all = all_showers[:,0]
    print('N of showers   Triggered   Trig. eff')
    print(len(energy_all), len(energy_trig), len(energy_trig)/len(energy_all))  # Trigger efficiency

    # Logaritmic binning in energy
    emin = 0.3  # TeV
    emax = 200
    etl_counts, etl_edges = np.histogram(energy_trig,bins=np.logspace(np.log10(emin),np.log10(emax), 100))
    eal_counts, eal_edges = np.histogram(energy_all,bins=np.logspace(np.log10(emin),np.log10(emax), 100))
    etl_centres = (etl_edges[:-1] + etl_edges[1:])/2.0
    eal_centres = (eal_edges[:-1] + eal_edges[1:])/2.0
    efficiencyl = etl_counts/eal_counts

    # Linear binning in energy
    et_counts, et_edges = np.histogram(energy_trig,bins=np.linspace(emin, emax, 100))
    ea_counts, ea_edges = np.histogram(energy_all,bins=np.linspace(emin, emax, 100))
    et_centres = (et_edges[:-1] + et_edges[1:])/2.0
    ea_centres = (ea_edges[:-1] + ea_edges[1:])/2.0
    efficiency = et_counts/ea_counts
    # replacing nan values by linear interpolation of neighbours
    ind = np.argwhere(np.isnan(efficiency))
    efficiency[ind] = (efficiency[ind-1] + efficiency[ind+1])/2.0
 
    
    # Effective area
    a_eff_log = efficiencyl*(np.pi*1000**2)  # Simple estimate according to D. Nedbal's disertation thesis
    a_eff = efficiency*(np.pi*1000**2)

    # Trigger rate
    
    # gamma spectrum (Crab at TeV energies)
    dNdE_gamma = 2.83*10**-7 * (et_centres)**-2.62  #[1/(s m^2 TeV)]
    dNdEs = a_eff*dNdE_gamma
    # integration over energy
    dE = et_edges[1] - et_edges[0]
    dNdt = sum(dNdEs*dE)
    print('Trigger rate [Hz]: ', 1.0/dNdt)

    # PLOTS
    fig = plt.figure(figsize=(10,8))
    plt.plot(etl_centres, etl_counts, 'r.')
    plt.plot(eal_centres, eal_counts, 'k.')
    plt.ylabel('Events')
    plt.xlabel('E [TeV]')
    plt.xscale('log')
    plt.yscale('log')

    fig = plt.figure(figsize=(10,8))
    plt.plot(etl_centres, efficiencyl, 'k.')
    plt.ylabel('trigger efficiency')
    plt.xlabel('E [TeV]')
    plt.xscale('log')
    plt.yscale('log')
    
    fig = plt.figure(figsize=(10,8))
    plt.plot(etl_centres, a_eff_log, 'k.')
    plt.ylabel('A eff [m^2]')
    plt.xlabel('E [TeV]')
    plt.xscale('log')
    plt.yscale('log')
    
    """
    fig = plt.figure(figsize=(10,8))
    plt.hist(np.log10(energy_trig), histtype='step', stacked=True, fill=False, linewidth=4, color='black', bins=100, range=[np.log10(.3), np.log10(200)])
    plt.hist(np.log10(energy_all), histtype='step', stacked=True, fill=False, linewidth=4, color='black', bins=100, range=[np.log10(.3), np.log10(200)])
    plt.yscale('log')
    """

    plt.show()
