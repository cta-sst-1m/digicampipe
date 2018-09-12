from optparse import OptionParser

import numpy as np

from digicampipe.image import hillas

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
            # data_cor2 = hillas.correct_alpha_1(data_cor, source_x=x, source_y=y)
            # data_cor2 = hillas.correct_alpha_2(data_cor, source_x=x, source_y=y)  # OK
            data_cor2 = hillas.correct_alpha_3(data_cor, source_x=x, source_y=y)  # OK
            # data_cor2 = hillas.correct_alpha_roland(data_cor,source_x=x, source_y=y)

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
