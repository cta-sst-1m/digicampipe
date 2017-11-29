from digicampipe.calib.camera import filter
from digicampipe.io.event_stream import event_stream
from digicampipe.io import save_adc
from digicampipe.calib.camera import r0
from digicampipe.io.save_bias_curve import save_bias_curve
from cts_core.camera import Camera
from digicampipe.utils import geometry
import matplotlib.pyplot as plt
import numpy as np
import utils.histogram as histogram
from digicampipe.utils import utils
from digicampipe.visualization import mpl as visualization
from cts_core import camera

from optparse import OptionParser

if __name__ == '__main__':
    #########################
    ##### CONFIGURATION #####
    #########################

    # Input configuration

    opts_parser = OptionParser()
    opts_parser.add_option('-d', '--directory', dest='directory', help='path', default='/home/alispach/data/CRAB_01/')
    opts_parser.add_option('-f', '--file', dest='filename', help='file basename', default='CRAB_01_0_000.%03d.fits.fz')
    opts_parser.add_option("-s", "--file_start", dest="file_start", help="file starting index", default=3, type=int)
    opts_parser.add_option("-e", "--file_end", dest="file_end", help="file starting index", default=23, type=int)
    opts_parser.add_option("-c", "--config_path", dest="config_path", help="config file path", default="/home/alispach/ctasoft/CTS/config/", type=str)

    (options, args) = opts_parser.parse_args()
    # Data configuration

    directory = options.directory
    filename = directory + options.filename
    file_list = [filename % number for number in range(options.file_start, options.file_end + 1)]
    camera_config_file = options.config_path + 'camera_config.cfg'
    pixel_histogram_filename = 'pixel_histogram.npz'
    patch_histogram_filename = 'patch_histogram.npz'
    cluster_7_histogram_filename = 'cluster_7_histogram.npz'
    cluster_19_histogram_filename = 'cluster_19_histogram.npz'
    trigger_filename = 'trigger.npz'
    display = True

    thresholds = np.arange(0, 400, 10)
    unwanted_patch = None  # [306, 318, 330, 342, 200]
    unwanted_cluster = None
    blinding = True

    pixel_histogram = histogram.Histogram(bin_center_min=0, bin_center_max=4095, bin_width=1, data_shape=(1296, ))
    patch_histogram = histogram.Histogram(bin_center_min=0, bin_center_max=255, bin_width=1, data_shape=(432, ))
    cluster_7_histogram = histogram.Histogram(bin_center_min=0, bin_center_max=1785, bin_width=1, data_shape=(432, ))
    cluster_19_histogram = histogram.Histogram(bin_center_min=0, bin_center_max=4845, bin_width=1, data_shape=(432, ))

    digicam = Camera(_config_file=camera_config_file)
    digicam_geometry = geometry.generate_geometry_from_camera(camera=digicam)

    # Define the event stream
    data_stream = event_stream(file_list=file_list, camera=digicam, expert_mode=True, camera_geometry=digicam_geometry)
    data_stream = filter.filter_event_types(data_stream, flags=[8])
    data_stream = filter.set_patches_to_zero(data_stream, unwanted_patch=unwanted_patch)
    data_stream = r0.fill_trigger_input_7(data_stream)
    data_stream = r0.fill_trigger_input_19(data_stream)
    # data_stream = r0.fill_trigger_input_offline(data_stream)
    # Fill the flags (to be replaced by Digicam)

    data_stream = save_bias_curve(data_stream, thresholds=thresholds, blinding=blinding, output_filename=directory + trigger_filename, unwanted_cluster=unwanted_cluster)

    data_stream = save_adc.fill_hist_adc_samples(data_stream, histogram=pixel_histogram, output_filename=directory + pixel_histogram_filename)
    data_stream = save_adc.fill_hist_trigger_input(data_stream, histogram=patch_histogram, output_filename=directory + patch_histogram_filename)
    data_stream = save_adc.fill_hist_cluster_7(data_stream, histogram=cluster_7_histogram, output_filename=directory + cluster_7_histogram_filename)
    data_stream = save_adc.fill_hist_cluster_19(data_stream, histogram=cluster_19_histogram, output_filename=directory + cluster_19_histogram_filename)

    if not display:

        for i, data in enumerate(data_stream):

            print(i)

    else:
        pixel_histogram = histogram.Histogram(filename=directory + pixel_histogram_filename)
        patch_histogram = histogram.Histogram(filename=directory + patch_histogram_filename)
        cluster_7_histogram = histogram.Histogram(filename=directory + cluster_7_histogram_filename)
        cluster_19_histogram = histogram.Histogram(filename=directory + cluster_19_histogram_filename)
        trigger = np.load(directory + trigger_filename)

        moments_pixel = utils.compute_moments(pixel_histogram.bin_centers, pixel_histogram.data)
        moments_patch = utils.compute_moments(patch_histogram.bin_centers, patch_histogram.data)
        moments_cluster_7 = utils.compute_moments(cluster_7_histogram.bin_centers, cluster_7_histogram.data)
        moments_cluster_19 = utils.compute_moments(cluster_19_histogram.bin_centers, cluster_19_histogram.data)

        np.savez('test.npz', **moments_pixel)
        mask = ~np.isnan(moments_pixel['mean'] * moments_pixel['std'] * moments_pixel['skewness'] * moments_pixel['kurtosis'])


        fig = plt.figure()
        axis = fig.add_subplot(111)
        geometry = geometry.generate_geometry_from_camera(camera=digicam)
        camera_visu = visualization.CameraDisplay(geometry, ax=axis, title='Kurtosis', cmap='viridis', allow_pick=True, norm='log')
        plt.axis('off')

        # if limits_colormap is not None:
        #    self.camera_visu.set_limits_minmax(limits_colormap[0], limits_colormap[1])

        # camera_visu.image = moments_pixel['mean']
        # plt.legend(loc='best')
        # plt.show()

        noisy_module = [1038, 1039, 1002, 1003, 1004, 966, 967, 968, 930, 931, 932, 896]

        mask = np.isnan(moments_pixel['kurtosis'])
        mask[noisy_module] = True
        camera_visu.image = np.ma.masked_array(moments_pixel['kurtosis'], mask=mask)
        camera_visu.cmap.set_bad('k')
        camera_visu.add_colorbar(orientation='horizontal', pad=0.03, fraction=0.05, shrink=.85)



        fig = plt.figure()
        axis = fig.add_subplot(111)
        camera_visu = visualization.CameraDisplay(geometry, ax=axis, title='Mean', cmap='viridis', allow_pick=True, norm='log')
        plt.axis('off')

        mask = np.isnan(moments_pixel['mean'])
        mask[noisy_module] = True
        camera_visu.image = np.ma.masked_array(moments_pixel['mean'], mask=mask)
        camera_visu.cmap.set_bad('k')
        camera_visu.add_colorbar(orientation='horizontal', pad=0.03, fraction=0.05, shrink=.85)

        fig = plt.figure()
        axis = fig.add_subplot(111)
        camera_visu = visualization.CameraDisplay(geometry, ax=axis, title='Std', cmap='viridis', allow_pick=True, norm='log')
        plt.axis('off')

        mask = np.isnan(moments_pixel['std'])
        mask[noisy_module] = True
        camera_visu.image = np.ma.masked_array(moments_pixel['std'], mask=mask)
        camera_visu.cmap.set_bad('k')
        camera_visu.add_colorbar(orientation='horizontal', pad=0.03, fraction=0.05, shrink=.85)

        fig = plt.figure()
        axis = fig.add_subplot(111)
        camera_visu = visualization.CameraDisplay(geometry, ax=axis, title='Skewness', cmap='viridis', allow_pick=True, norm='log')
        plt.axis('off')

        mask = np.isnan(moments_pixel['skewness'])
        mask[noisy_module] = True
        camera_visu.image = np.ma.masked_array(moments_pixel['skewness'], mask=mask)
        camera_visu.cmap.set_bad('k')
        camera_visu.add_colorbar(orientation='horizontal', pad=0.03, fraction=0.05, shrink=.85)

        plt.show()

        mask = ~np.isnan(moments_pixel['mean'] * moments_pixel['std'] * moments_pixel['skewness'] * moments_pixel['kurtosis'])
        plt.figure()
        plt.hist(moments_pixel['mean'][mask], label='mean', bins='auto')
        plt.legend()

        plt.figure()
        plt.hist(moments_pixel['std'][mask], label='std', bins='auto')
        plt.legend()

        plt.figure()
        plt.hist(moments_pixel['skewness'][mask], label='skewness', bins='auto')
        plt.legend()

        plt.figure()
        plt.hist(moments_pixel['kurtosis'][mask], label='kurtosis', bins='auto')
        plt.legend()
        plt.show()

        directory = directory + 'figures/'

        for i in range(trigger['threshold'].shape[0]):

            fig = plt.figure()
            axis = fig.add_subplot(111)
            n_entries = np.sum(pixel_histogram.data[0])
            period = n_entries * 4
            x = np.arange(cluster_7_histogram.data.shape[0])
            width = 1
            x = x - width / 3
            y = trigger['cluster_rate'][:, i] * period
            yerr = trigger['cluster_rate_error'][:, i] * period
            axis.bar(x, y, width, label=' threshold : {} [LSB]\n total : {}'.format(trigger['threshold'][i], np.sum(y)))
            axis.set_xlabel('cluster 7 ID')
            axis.legend()
            fig.savefig(directory + 'trigger_count_threshold_{}.svg'.format(trigger['threshold'][i]))
            plt.close()

        fig = plt.figure()
        axis = fig.add_subplot(111)
        axis.errorbar(trigger['threshold'], trigger['rate'] * 1E9, yerr=trigger['rate_error'] * 1E9,
                      label='Blinding : {}'.format(blinding))
        axis.set_ylabel('rate [Hz]')
        axis.set_xlabel('threshold [LSB]')
        axis.set_yscale('log')
        axis.legend(loc='best')
        fig.savefig(directory + 'bias_curve.svg')
        plt.close()

        for i in range(pixel_histogram.data.shape[0]):

            fig = plt.figure()
            axis = fig.add_subplot(111)
            mask = pixel_histogram.data[i] > 0
            mean = moments_pixel['mean'][i]
            std = moments_pixel['std'][i]
            skewness = moments_pixel['skewness'][i]
            kurtosis = moments_pixel['kurtosis'][i]
            n_entries = moments_pixel['n_entries'][i]
            label = ' pixel : {}\n mean : {:0.2f} [LSB]\n std : {:0.2f} [LSB]\n skewness : {:0.2f} []\n kurtosis : {:0.2f} []\n entries : {}'.format(
                i, mean, std, skewness, kurtosis, n_entries)
            axis.step(pixel_histogram.bin_centers[mask], pixel_histogram.data[i][mask], label=label, where='mid')
            axis.set_xlabel('[LSB]')
            axis.legend()
            fig.savefig(directory + 'pixel_{}.svg'.format(i))
            plt.close()

        0/0

        for i in range(patch_histogram.data.shape[0]):

            fig = plt.figure()
            axis = fig.add_subplot(111)
            mask = patch_histogram.data[i] > 0
            x = patch_histogram.bin_centers[mask]
            y = patch_histogram.data[i][mask]
            mean = np.average(x, weights=y)
            std = np.average((x - mean)**2, weights=y)
            std = np.sqrt(std)
            n_entries = np.sum(y)
            skewness = np.average(((x - mean)/std)**3, weights=y)
            kurtosis = np.average(((x - mean)/std)**4, weights=y)
            label = ' patch : {}\n mean : {:0.2f} [LSB]\n std : {:0.2f} [LSB]\n skewness : {:0.2f} []\n kurtosis : {:0.2f} [] \n entries : {}'.format(i, mean, std, skewness, kurtosis, n_entries)
            axis.step(x, y, label=label, where='mid')
            axis.set_xlabel('[LSB]')
            axis.legend()
            fig.savefig(directory + 'cluster_7_{}.svg'.format(i))
            plt.close()

            fig = plt.figure()
            axis = fig.add_subplot(111)
            y = trigger['cluster_rate'][i] * 1E9
            y_err = trigger['cluster_rate_error'][i] * 1E9
            x = trigger['threshold']
            axis.errorbar(x, y, yerr=y_err, label=' cluster 7 : {}'.format(i))
            axis.set_xlabel('threshold [LSB]')
            axis.set_yscale('log')
            axis.set_ylabel('rate [Hz]')
            axis.legend()
            fig.savefig(directory + 'trigger_cluster_7_{}.svg'.format(i))
            plt.close()

        for i in range(cluster_7_histogram.data.shape[0]):

            fig = plt.figure()
            axis = fig.add_subplot(111)
            mask = cluster_7_histogram.data[i] > 0
            x = cluster_7_histogram.bin_centers[mask]
            y = cluster_7_histogram.data[i][mask]
            mean = np.average(x, weights=y)
            std = np.average((x - mean)**2, weights=y)
            std = np.sqrt(std)
            n_entries = np.sum(y)
            label = ' cluster 7 : {}\n mean : {:0.2f} [LSB]\n std : {:0.2f} [LSB]\n skewness : {:0.2f} []\n kurtosis : {:0.2f} []\n entries : {}'.format(i, mean, std, skewness, kurtosis, n_entries)
            axis.step(x, y, label=label, where='mid')
            axis.set_xlabel('[LSB]')
            axis.legend()
            fig.savefig(directory + 'cluster_7_{}.svg'.format(i))
            plt.close()

        for i in range(cluster_19_histogram.data.shape[0]):

            fig = plt.figure()
            axis = fig.add_subplot(111)
            mask = cluster_19_histogram.data[i] > 0
            x = cluster_19_histogram.bin_centers[mask]
            y = cluster_19_histogram.data[i][mask]
            mean = np.average(x, weights=y)
            std = np.average((x - mean)**2, weights=y)
            std = np.sqrt(std)
            n_entries = np.sum(y)
            label = ' cluster_19 : {}\n mean : {:0.2f} [LSB]\n std : {:0.2f} [LSB]\n skewness : {:0.2f} []\n kurtosis : {:0.2f} []\n entries : {}'.format(i, mean, std, skewness, kurtosis, n_entries)
            axis.step(x, y, label=label, where='mid')
            axis.set_xlabel('[LSB]')
            axis.legend()
            fig.savefig(directory + 'cluster_19_{}.svg'.format(i))
            plt.close()


