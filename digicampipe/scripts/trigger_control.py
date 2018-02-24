'''
Usage:
  dg_trigger_control [options] <files>

Options:
  -h, --help  Show this help
  -o DIR, --out DIR   output directory
  --unblind     Do not use blinding
  --by_patch    Dunno
'''
import matplotlib.pyplot as plt
import numpy as np
from docopt import docopt
from tqdm import tqdm

from digicampipe.io import event_stream
from digicampipe.io import save_adc
from digicampipe.io import save_bias_curve
from digicampipe.calib.camera import r0, filter
import utils.histogram as histogram
from functools import partial


def entry():
    args = docopt(__doc__)

    thresholds = np.arange(0, 400, 10)
    unwanted_patch = None
    unwanted_cluster = None
    blinding = not args['--unblind']
    by_cluster = not args['--by_patch']

    r0_histograms = {
        'adc_samples': {'nbins': 1296},
        'trigger_input_traces': {'nbins': 432},
        'trigger_input_7': {'nbins': 432},
        'trigger_input_19': {'nbins': 432},
    }
    r0_histogram_fillers = {
        fieldname: save_adc.R0HistogramFiller(fieldname, **args)
        for fieldname, args in r0_histograms.items()
    }

    process = [
        partial(filter.filter_event_types, flags=[8]),
        partial(filter.set_patches_to_zero, unwanted_patch=unwanted_patch),
        r0.fill_trigger_input_7,
        r0.fill_trigger_input_19,
        save_bias_curve.RateThresholdAnalysis(
            thresholds,
            blinding=blinding,
            by_cluster=by_cluster,
            unwanted_cluster=unwanted_cluster
        ),
        *r0_histogram_fillers.values()
    ]

    data_stream = event_stream(args['<files>'])
    for processor in process:
        data_stream = processor(data_stream)
    for _ in tqdm(data_stream):
        pass


def plot():
    # THis dows not work yet
    pixel_histogram = histogram.Histogram(filename=directory + pixel_histogram_filename)
    patch_histogram = histogram.Histogram(filename=directory + patch_histogram_filename)
    cluster_7_histogram = histogram.Histogram(filename=directory + cluster_7_histogram_filename)
    cluster_19_histogram = histogram.Histogram(filename=directory + cluster_19_histogram_filename)
    trigger = np.load(directory + trigger_filename)

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
        mean = np.average(pixel_histogram.bin_centers, weights=pixel_histogram.data[i])
        std = np.average((pixel_histogram.bin_centers - mean)**2, weights=pixel_histogram.data[i])
        std = np.sqrt(std)
        skewness = np.average(((pixel_histogram.bin_centers - mean)/std)**3, weights=pixel_histogram.data[i])
        kurtosis = np.average(((pixel_histogram.bin_centers - mean)/std)**4, weights=pixel_histogram.data[i])
        n_entries = np.sum(pixel_histogram.data[i][mask])
        label = ' pixel : {}\n mean : {:0.2f} [LSB]\n std : {:0.2f} [LSB]\n skewness : {:0.2f} []\n kurtosis : {:0.2f} []\n entries : {}'.format(
            i, mean, std, skewness, kurtosis, n_entries)
        axis.step(pixel_histogram.bin_centers[mask], pixel_histogram.data[i][mask], label=label, where='mid')
        axis.set_xlabel('[LSB]')
        axis.legend()
        fig.savefig(directory + 'pixel_{}.svg'.format(i))
        plt.close()

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
