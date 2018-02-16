from pyhessio import open_hessio
import numpy as np
from optparse import OptionParser
import os


def hessio_all_source(url):
    
    with open_hessio(url) as pyhessio:

        eventstream = pyhessio.move_to_next_mc_event()
        event_id0 = 0
        for event_id in eventstream:

            energy = pyhessio.get_mc_shower_energy()
            theta = 90 - np.rad2deg(pyhessio.get_mc_shower_altitude())
            phi = np.rad2deg(pyhessio.get_mc_shower_azimuth())
            x_core = pyhessio.get_mc_event_xcore()
            y_core = pyhessio.get_mc_event_ycore()
            h_first_int = pyhessio.get_mc_shower_h_first_int()
            event_id = pyhessio.get_global_event_count()

            if event_id != event_id0:   # trigger flag
                trig = 1
            else: trig = 0
            event_id0 = event_id

            mc_data = np.hstack((energy, theta, phi, h_first_int, x_core, y_core, trig))

            yield mc_data


def all_showers_stream(file_list):
    
    for file in file_list:

        data_stream = hessio_all_source(file)

        for event in data_stream:

            yield event


def save_file(mc_data_all, filename_showerparam):

    np.savetxt(filename_showerparam, mc_data_all, '%1.5f %1.1f %1.1f %1.5f %1.5f %1.5f %d')


if __name__ == '__main__':

    parser = OptionParser()
    parser.add_option("-p", "--path", dest="directory", help="directory to data files",
                      default='../../../sst-1m_simulace/data_test/ryzen_testprod/0.0deg/Data/')
    parser.add_option('-z', "--zenit", dest='zenit_angle', help='Zenit distance, THETAP', default=0, type=int)
    parser.add_option('-a', "--azimut", dest='azimut', help='Azimut, PHIP', default=0, type=int)
    parser.add_option('-r', "--primary", dest='primary', help='Primary particle', default='gamma', type=str)
    (options, args) = parser.parse_args()


    # Input/Output files
    directory = options.directory
    all_file_list = os.listdir(directory)
    file_list = []
    string1 = options.primary + '_' + str(options.zenit_angle) + 'deg_' + str(options.azimut) + 'deg_'

    for fi in all_file_list:
        if string1 in fi and '___cta-prod3-sst-dc-2150m--sst-dc' in fi and '.simtel.gz' in fi:
            print(fi)
            file_list.append(directory + fi)

    data_stream = all_showers_stream(file_list=file_list)

    mc_data = []
    for shower in data_stream:

        mc_data.append(shower)

    mc_data = np.array(mc_data)

    print('N of showers: ', len(mc_data))
    print('Triggered: ', len(mc_data[mc_data[:,6] == 1,:]))
    
    filename_showerparam = 'shower_allmc_param_' + options.primary + '_ze' + str(options.zenit_angle).zfill(2) + '_az' + str(options.azimut).zfill(3) + '.txt'
    save_file(mc_data, options.directory + filename_showerparam)  # save mc parameters of all showers











