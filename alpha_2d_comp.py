
import numpy as np
import sys
import matplotlib.pyplot as plt
from optparse import OptionParser


def plot_alpha(datas, **kwargs):

    mask = ~np.isnan(datas['alpha']) * ~np.isinf(datas['alpha'])
    fig = plt.figure(figsize=(14, 12))
    ax1 = fig.add_subplot(111)
    ax1.set_ylabel('alpha [deg]')
    ax1.hist(datas['alpha'][mask], bins='auto', **kwargs)


def correct_alpha(datas, source_x, source_y):  # cyril
    """
    datas['cen_x'] = datas['cen_x'] - source_x
    datas['cen_y'] = datas['cen_y'] - source_y
    datas['r'] = np.sqrt((datas['cen_x'])**2 + (datas['cen_y'])**2)
    datas['phi'] = np.arctan2(datas['cen_y'], datas['cen_x'])
    datas['alpha'] = np.sin(datas['phi']) * np.sin(datas['psi']) + np.cos(datas['phi']) * np.cos(datas['psi'])
    datas['alpha'] = np.arccos(datas['alpha'])
    # data['alpha'] = np.abs(data['phi'] - data['psi'])
    datas['alpha'] = np.remainder(datas['alpha'], np.pi/2)
    datas['alpha'] = np.rad2deg(datas['alpha'])    # conversion to degrees
    datas['miss'] = datas['r'] * np.sin(datas['alpha'])
    return datas
    """

    xx = datas['cen_x'] - source_x
    yy = datas['cen_y'] - source_y
    datas['r'] = np.sqrt(xx**2.0 + yy**2.0)
    datas['phi'] = np.arctan2(yy, xx)
    datas['alpha'] = np.arccos(np.sin(datas['phi']) * np.sin(datas['psi']) + np.cos(datas['phi']) * np.cos(datas['psi']))
    datas['alpha'] = np.remainder(datas['alpha'], np.pi/2)
    datas['alpha'] = np.rad2deg(datas['alpha'])    # conversion to degrees
    datas['miss'] = datas['r'] * np.sin(datas['alpha'])
    return datas


def alpha_cyril(datas, source_x, source_y):  # cyril from prod_alpha_plot.c

    x = datas['cen_x'] - source_x
    y = datas['cen_y'] - source_y
    datas['r'] = np.sqrt(x**2.0 + y**2.0)
    phi = np.arctan(y/x)
    calpha = np.sin(phi) * np.sin(datas['psi']) + np.cos(phi) * np.cos(datas['psi'])
    alpha = np.arccos(calpha)
    for i in range(len(alpha)):
        if alpha[i] > np.pi/2.0:
            alpha[i] = np.pi - alpha[i]
    # delta_alpha=np.arctan2(source_y,source_x)-np.arctan2(-1.0*datas['cen_y'],-1.0*datas['cen_x']);
    # alpha_r = alpha + delta_alpha;
    # miss_c = r*TMath::Sin(alpha_c);
    # miss_r = r*TMath::Sin(alpha_r);
    datas['alpha'] = alpha
    datas['alpha'] = np.rad2deg(datas['alpha'])    # conversion to degrees
    datas['miss'] = datas['r'] * np.sin(datas['alpha'])
    return datas


"""
def alpha_roland(datas, source_x, source_y): #roland from prod_alpha_plot.c

    alpha = np.sin(datas['miss']/datas['r'])
    alpha2 = np.arctan2(-datas['cen_y'], -datas['cen_x']) - datas['psi'] + np.pi

    for i in range(len(alpha2)):
        if (alpha2[i] > np.pi):
            alpha2[i] = alpha2[i] - 2*np.pi
        elif (alpha2[i] < -np.pi):
            alpha2[i] = alpha2[i] + 2*np.pi

    delta_alpha2 = np.arctan2(source_y-datas['cen_y'],source_x-datas['cen_x'])-np.arctan2(-datas['cen_y'], -datas['cen_x'])
    alpha2_crab = alpha2 + delta_alpha2

    for i in range(len(alpha2_crab)):
        if (alpha2_crab[i] > 2*np.pi):
            alpha2_crab[i] = alpha2_crab[i] - 2*np.pi
        elif (alpha2_crab[i] < -2*np.pi):
            alpha2_crab[i] = alpha2_crab[i] + 2*np.pi

    for i in range(len(alpha2_crab)):
        if (alpha2_crab[i] > np.pi):
            alpha2_crab[i] = 2*np.pi - alpha2_crab[i]
        elif (alpha2_crab[i] < -np.pi):
            alpha2_crab[i] = -2*np.pi - alpha2_crab[i]

    alpha2_crab = abs(alpha2_crab)

    for i in range(len(alpha2_crab)):
        if (alpha2_crab[i] > 0.5*np.pi):
            alpha2_crab[i] = np.pi - alpha2_crab[i]

    datas['alpha'] = alpha2_crab
    return datas
"""

def alpha_etienne(datas, source_x, source_y):  # etienne's code from scan_crab_cluster.c


    d_x = np.cos(datas['psi'])
    d_y = np.sin(datas['psi'])
    to_c_x = source_x - datas['cen_x']
    to_c_y = source_y - datas['cen_y']
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
    datas['alpha'] = 180.0/np.pi*alpha_cetienne
    datas['r'] = to_c_norm
    datas['miss'] = datas['r'] * np.sin(datas['alpha'])
    return datas


if __name__ == '__main__':

    parser = OptionParser()
    parser.add_option("-p", "--path", dest="path", help="path to data files", default='../../sst-1m_data/20171030/output_crab_dark1_18_noaddrow_minphot80.npz')
    parser.add_option("-o", "--output", dest="output", help="output filename", default="alpha_2d", type=str)
    parser.add_option("-b", "--bining", dest="bining", help="size of bins in alpha histogram [degrees]", default=4, type=float)
    parser.add_option("-s", "--steps", dest="steps", help="number of steps, resolution of 2d alpha plot", default=100, type=int)
    parser.add_option("-l", "--lower_cut", dest="lower_cut", help="lower cut on length/width ratio", default=1.5, type=float)
    parser.add_option("-u", "--upper_cut", dest="upper_cut", help="upper cut on length/width ratio", default=3, type=float)
    (options, args) = parser.parse_args()

    # settings
    lw_min = options.lower_cut
    lw_max = options.upper_cut
    bin_size = options.bining  # degrees
    num_steps = options.steps
    output_filename = options.output

    # data loading
    data = np.load(options.path)
    mask = np.ones(data['size'].shape[0], dtype=bool)

    # application of cuts
    mask = (data['size'] > 0) & (data['length']/data['width'] > lw_min) & (data['length']/data['width'] < lw_max) & (data['border'] == 0)

    data_cor = dict()
    for key, val in data.items():
        data_cor[key] = val[mask]

    x_pos, y_pos, N = [], [], []

    x_crab_start = -500
    y_crab_start = -500
    x_crab_end = 500
    y_crab_end = 500

    x_crab = np.linspace(x_crab_start, x_crab_end, num_steps)
    y_crab = np.linspace(y_crab_start, y_crab_end, num_steps)

    # print(min(data['cen_x']), max(data['cen_x']))
    # print(min(data['cen_y']), max(data['cen_y']))
    # x_crab_centre = 40
    # y_crab_centre = 13.5

    i = 0
    for x in x_crab:
        print(round(i/len(x_crab)*100, 2), '/', 100)  # progress
        for y in y_crab:
            # x = x_crab_centre
            # y = y_crab_centre

            # alpha computing
            # data_cor2 = correct_alpha(data_cor, source_x=x, source_y=y)
            # data_cor2 = alpha_cyril(data_cor, source_x=x, source_y=y)  # OK
            data_cor2 = alpha_etienne(data_cor, source_x=x, source_y=y)  # OK
            # data_cor2 = alpha_roland(data_cor,source_x=x, source_y=y)

            # 'first bin' sellection + simplified r-min criterion (source musn't be inside the elipse)
            # - with use of the second criterion, number of events in pixel differs from the case without the criterion only about 1% max..
            # - plots look the same
            data_cor_bin = dict()
            mask2 = (data_cor2['alpha'] < bin_size) & (data_cor2['r'] - data_cor2['length']/2.0 > 0)
            # mask3 = (data_cor2['alpha'] < bin_size)
            # if abs((len(mask2[mask2 == True])-len(mask3[mask3 == True]))/len(mask3[mask3 == True]))>0.01:
            # 	print((len(mask2[mask2 == True])-len(mask3[mask3 == True]))/len(mask3[mask3 == True]))
            for key, val in data_cor2.items():
                data_cor_bin[key] = val[mask2]

            # output lists
            x_pos.append(x)
            y_pos.append(y)
            N.append(data_cor_bin['alpha'].shape[0])

            # plot_alpha(data_cor2)
            # plt.show()
        i += 1

    # save output
    np.savez(output_filename, x=x_pos, y=y_pos, N=N)  # save to npz
    np.savetxt(output_filename+'.txt', np.transpose([x_pos, y_pos, N]), fmt='%1.3f %1.3f %d') # save to txt
