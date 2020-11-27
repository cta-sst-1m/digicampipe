import numpy as np
import matplotlib.pyplot as plt
from digicampipe.scripts.all_in_one_reco import get_data
import argparse
import os
from digicampipe.visualization.machine_learning import plot_features, plot_weighted_features, LABELS
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import LogNorm
from tqdm import tqdm


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some value.')
    parser.add_argument('--max_events', default=None, type=int,
                        help='sum the integers (default: find the max)')
    parser.add_argument('--proton',
                        default='/sst1m/analyzed/mpeshowerfit/tmp/zenith_20_proton_v15.hdf5',
                        type=str, help='Processing version')
    parser.add_argument('--gamma',
                        default='/sst1m/analyzed/mpeshowerfit/tmp/zenith_20_gamma_v15.hdf5',
                        type=str, help='Processing version')
    parser.add_argument('--gamma_diffuse',
                        default='/sst1m/analyzed/mpeshowerfit/tmp/zenith_20_diffuse_gamma_v15.hdf5',
                        type=str, help='Processing version')
    parser.add_argument('--output', default='./dl1_parameters.pdf',
                        type=str, help='Processing version')

    parser.add_argument('--kind', default='fit',
                        type=str, help='use the fit values or the init values')
    args = parser.parse_args()


    max_events = args.max_events
    kind = args.kind
    output_filename = args.output
    gamma_file = args.gamma
    proton_file = args.proton
    gamma_diffuse_file = args.gamma_diffuse
    size_cut = 0
    leakage_cut = (0, np.inf)
    wol_cut = 0

    gamma_counts, gamma_counts_triggered, gamma_counts_fitted, df_gamma_quality, gamma_bins = get_data(
        file=gamma_file, max_events=max_events, intensity_cut=size_cut, leakage_cut=leakage_cut, wol_cut=wol_cut,
        kind=kind)

    print(df_gamma_quality['weights'])
    print('Total rate after quality cut gamma {:.2f} [mHz]'.format(
        np.sum(df_gamma_quality['weights']) * 1E3))
    print("Reading proton sample")
    proton_counts, proton_counts_triggered, proton_counts_fitted, df_proton_quality, proton_bins = get_data(
        file=proton_file, max_events=max_events, intensity_cut=size_cut, leakage_cut=leakage_cut, wol_cut=wol_cut,
        kind=kind)

    print('Total rate after quality cut proton {:.2f} [Hz]'.format(
        np.sum(df_proton_quality['weights'])))
    print("Reading gamma diffuse sample")
    gamma_diffuse_counts, gamma_diffuse_counts_triggered, gamma_diffuse_counts_fitted, df_gamma_diffuse_quality, gamma_bins = get_data(
        file=gamma_diffuse_file, max_events=max_events, intensity_cut=size_cut, leakage_cut=leakage_cut, wol_cut=wol_cut,
        kind=kind)

    features = df_gamma_quality.columns
    target_features = ['impact', 'true_energy', 'x_max', 'disp_r', 'disp_theta']

    with PdfPages(output_filename) as pdf:

        for feature in tqdm(features):

            fig = plt.figure()
            axes = plot_features(feature=feature, df_gamma=df_gamma_quality,
                                 df_proton=df_proton_quality,
                                 df_gamma_diffuse=df_gamma_diffuse_quality)
            pdf.savefig(axes.get_figure())

            fig = plt.figure()
            axes = plot_weighted_features(feature=feature,
                                          df_gamma=df_gamma_quality,
                                          df_proton=df_proton_quality,)
            pdf.savefig(axes.get_figure())

            for target in target_features:

                x = np.array(df_gamma_quality[target])
                y = np.array(df_gamma_quality[feature])
                mask = np.isfinite(x) & np.isfinite(y)
                x, y = x[mask], y[mask]
                bins_x = np.linspace(x.min(), x.max(), num=100)
                bins_y = np.linspace(y.min(), y.max(), num=100)
                bins = [bins_x, bins_y]

                fig = plt.figure()
                axes = fig.add_subplot(111)
                H = axes.hist2d(x, y, bins=bins, norm=LogNorm())
                axes.set_xlabel(LABELS[target])
                axes.set_ylabel(LABELS[feature])
                fig.colorbar(H[3], label='count []', ax=axes)
                pdf.savefig(fig)