import numpy as np
import matplotlib.pyplot as plt
from digicampipe.visualization.machine_learning import plot_feature_importance, plot_confusion_matrix
import glob
import os
import pickle as pk
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import roc_curve, accuracy_score, roc_auc_score, \
    confusion_matrix, recall_score, r2_score
from sklearn.model_selection import \
    GridSearchCV  # Create the parameter grid based on the results of random search

from digicampipe.io.dl1 import create_data_frame, read_hdf5, combine_datasets, quality_cuts
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import LogNorm


def plot_hist_2d(x, y, bins=(50, 50), axes=None, line=True, **kwargs):
    if axes is None:
        fig = plt.figure()
        axes = fig.add_subplot(111)

    mask = np.isfinite(x) * np.isfinite(y)
    x = x[mask]
    y = y[mask]

    if isinstance(bins, tuple):

        x_bins = np.linspace(x.min(), x.max(), num=bins[0])
        y_bins = np.linspace(y.min(), y.max(), num=bins[1])
    else:
        x_bins = bins[0]
        y_bins = bins[1]

    H = axes.hist2d(x, y, bins=[x_bins, y_bins], **kwargs)

    if line:
        count = H[0]
        x_bins = H[1][:-1] + np.diff(H[1]) / 2.
        y_bins = H[2][:-1] + np.diff(H[2]) / 2.

        radius = np.zeros(len(x_bins))

        for i in range(len(H[1]) - 1):
            mask = np.isfinite(x) * np.isfinite(y) * (x <= H[1][i + 1]) * (
                        x > H[1][i])
            data = y[mask]
            radius[i] = np.quantile(data, 0.68)

            # mean_y= (y_bins * count).sum(axis=1) / count.sum(axis=1)
        # axes.plot(x_bins, mean_y, label='mean', color='r')

        axes.plot(x_bins, radius, label='68 %', color='r')
    plt.colorbar(H[3], label='count []')
    axes.legend(loc='best')

    return axes


test_size = 0.2
max_files = None
n_estimators = 300
# input_dir = '/data/sst-1m_prod4b/'
input_dir = '/work/se2/alispach/sst-1m_prod4b/'
output_dir = '/work/se2/alispach/sst-1m_prod4b/'

gamma_file = os.path.join(input_dir, 'gamma/gamma_v2.hdf5')
df_gamma = read_hdf5(gamma_file)
df_gamma = quality_cuts(df_gamma)
df_gamma_train, df_gamma_test = train_test_split(df_gamma, test_size=test_size)

df_test = df_gamma_test
df_train = df_gamma_train

print("Test sample contains {:d} events\n".format(len(df_test)))
print("Train sample contains {:d} events\n".format(len(df_train)))

dropped_features = ['true_energy', 'event_id', 'particle',
                    'tel_id', 'valid', 'intercept',
                    'x', 'y', 'r', 'x_err', 'y_err', 'r_err',
                    'leakage', 'alt', 'az', 'tel_alt', 'tel_az']

df_train_X, df_train_y = df_train.drop(dropped_features, axis=1), df_train['true_energy']
df_test_X, df_test_y = df_test.drop(dropped_features, axis=1), df_test['true_energy']
features = df_train_X.columns

print("Training energy regressor with\n {}".format(features))

rf_energy = RandomForestRegressor(n_estimators=n_estimators, n_jobs=-1, verbose=3)
rf_energy.fit(df_train_X, df_train_y)

reco_energy = rf_energy.predict(df_test_X)
r2_test = r2_score(df_test_y, reco_energy)
r2_train = r2_score(df_train_y, rf_energy.predict(df_train_X))
importance = rf_energy.feature_importances_

print('Test R^2 score {:.6f}'.format(r2_test))
print('Train R^2 score {:.6f}'.format(r2_train))

n_x, n_y = 100, 100

with PdfPages(os.path.join(output_dir,'figure_energy_regressor.pdf')) as pdf:

    axes = plot_feature_importance(features, importance)
    pdf.savefig(axes.get_figure())

    x_bins = np.linspace(np.log10(df_test_y).min(), np.log10(df_test_y).max(), num=n_x)
    y_bins = np.linspace(-2, 2, num=n_y)

    axes = plot_hist_2d(np.log10(df_test_y), (reco_energy - df_test_y) / df_test_y,
                        bins=[x_bins, y_bins],
                        norm=LogNorm(), line=False)
    axes.set_xlabel(r'True $\log_{10}\left(\frac{E}{{\rm TeV}}\right)$')
    axes.set_ylabel(r'$\frac{E_{reco} - E_{true}}{E_{true}}$ []')
    pdf.savefig(axes.get_figure())

    diff_reco = np.zeros(len(x_bins) - 1) * np.nan
    diff_reco_median = np.zeros(len(x_bins) - 1) * np.nan
    diff_68_reco = np.zeros(len(x_bins) - 1) * np.nan
    diff_68_reco_jakub = np.zeros(len(x_bins) - 1) * np.nan
    diff_reco_std = np.zeros(len(x_bins) - 1) * np.nan
    x_centers = np.zeros(len(x_bins) - 1) * np.nan
    x_err = np.zeros(len(x_bins) - 1) * np.nan

    for i in range(len(x_bins) - 1):
        mask = (np.log10(df_test_y) <= x_bins[i + 1]) * (np.log10(df_test_y) > x_bins[i])

        if mask.sum() > 100:

            diff = (reco_energy[mask] - df_test_y[mask]) / df_test_y[mask]
            diff_reco[i] = np.mean(diff)
            diff_reco_median[i] = np.median(diff)
            diff_reco_std[i] = np.std(diff)
            diff_68_reco[i] = np.quantile(np.abs(diff), 0.682)
            diff_68_reco_jakub[i] = -(np.quantile(diff, 0.15865) - np.quantile(diff, 0.84135)) * 0.5
            x_err[i] = (x_bins[i + 1] - x_bins[i]) / 2
            # x_centers[i] = np.mean(df_test_y[mask])
            x_centers[i] = x_bins[i] + x_err[i]
            # x_centers[i] = np.log10(x_centers[i])
            # x_err[i] = np.log10(x_err[i])

    fig = plt.figure()
    axes = fig.add_subplot(111)
    axes.plot(x_centers, diff_68_reco, color='r', linestyle='--', label='68.2 %')
    axes.plot(x_centers, diff_68_reco_jakub, color='g', linestyle='--', label='Jakub 68.2 %')
    # axes.plot(x_centers, diff_reco_std, color='g', linestyle='--', label='Std')
    axes.plot(x_centers, diff_reco, color='k', label='Mean')
    axes.plot(x_centers, diff_reco_median, color='b', label='Median')
    axes.set_xlabel(r'True $\log_{10}\left(\frac{E}{{\rm TeV}}\right)$')
    axes.set_ylabel(r'$\frac{E_{reco} - E_{true}}{E_{true}}$ []')
    axes.set_ylim((-0.5, 0.5))
    axes.legend(loc='best')
    axes.set_ylim(None, 1)
    pdf.savefig(fig)

    x_bins = np.linspace(np.log10(reco_energy).min(), np.log10(reco_energy).max(),
                         num=n_x)
    y_bins = np.linspace(-2, 2, num=n_y)

    axes = plot_hist_2d(np.log10(reco_energy),
                        (reco_energy - df_test_y) / df_test_y,
                        bins=[x_bins, y_bins],
                        norm=LogNorm(), line=False)
    axes.set_xlabel(r'Reco $\log_{10}\left(\frac{E}{{\rm TeV}}\right)$')
    axes.set_ylabel(r'$\frac{E_{reco} - E_{true}}{E_{true}}$ []')
    pdf.savefig(axes.get_figure())

    diff_reco = np.zeros(len(x_bins) - 1) * np.nan
    diff_reco_median = np.zeros(len(x_bins) - 1) * np.nan
    diff_68_reco = np.zeros(len(x_bins) - 1) * np.nan
    diff_reco_std = np.zeros(len(x_bins) - 1) * np.nan
    x_centers = np.zeros(len(x_bins) - 1) * np.nan
    x_err = np.zeros(len(x_bins) - 1) * np.nan

    for i in range(len(x_bins) - 1):
        mask = (np.log10(reco_energy) <= x_bins[i + 1]) * (
                    np.log10(reco_energy) > x_bins[i])

        if mask.sum() > 100:
            diff = (reco_energy[mask] - df_test_y[mask]) / df_test_y[mask]
            diff_reco[i] = np.mean(diff)
            diff_reco_median[i] = np.median(diff)
            diff_68_reco[i] = np.quantile(np.abs(diff), 0.682)
            diff_reco_std[i] = np.std(diff)
            x_err[i] = (x_bins[i + 1] - x_bins[i]) / 2
            # x_centers[i] = np.mean(df_test_y[mask])
            x_centers[i] = x_bins[i] + x_err[i]
            # x_centers[i] = np.log10(x_centers[i])
            # x_err[i] = np.log10(x_err[i])

    fig = plt.figure()
    axes = fig.add_subplot(111)
    axes.plot(x_centers, diff_68_reco, color='r', linestyle='--', label='68.2 %')
    # axes.plot(x_centers, diff_reco_std, color='g', linestyle='--', label='Std')
    axes.plot(x_centers, diff_reco, color='k', label='Mean')
    axes.plot(x_centers, diff_reco_median, color='b', label='Median')
    axes.set_xlabel(r'Reco $\log_{10}\left(\frac{E}{{\rm TeV}}\right)$')
    axes.set_ylabel(r'$\frac{E_{reco} - E_{true}}{E_{true}}$ []')
    axes.legend(loc='best')
    axes.set_ylim((-0.5, 0.5))
    pdf.savefig(fig)

    plt.show()

output_filename = os.path.join(output_dir, 'energy_regressor.pk')

with open(output_filename, 'wb') as f:

     pk.dump(rf_energy, f)
