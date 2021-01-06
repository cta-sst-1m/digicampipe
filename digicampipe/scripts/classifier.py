
import numpy as np
import matplotlib.pyplot as plt
from digicampipe.visualization.machine_learning import plot_feature_importance, plot_confusion_matrix
import os
import pickle as pk
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, accuracy_score, roc_auc_score, \
    confusion_matrix, recall_score
from sklearn.model_selection import \
    GridSearchCV  # Create the parameter grid based on the results of random search

from digicampipe.io.dl1 import read_hdf5, combine_datasets, quality_cuts
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd

test_size = 0.2
max_events = None
size_cut = 160

# input_dir = '/work/se2/alispach/sst-1m_prod4b/'
input_dir = '/sst1m/analyzed/mpeshowerfit/'
# output_dir = '/work/se2/alispach/sst-1m_prod4b/'
output_dir = '/sst1m/analyzed/mpeshowerfit/'

gamma_basename = 'zenith_20_gamma_v13.hdf5'
proton_basename = 'zenith_20_proton_v13.hdf5'
gamma_diffuse_basename = 'zenith_20_diffuse_gamma_v13.hdf5'
gamma_files = os.path.join(input_dir, gamma_basename)
proton_files = os.path.join(input_dir, proton_basename)
gamma_diffuse_files = os.path.join(input_dir, gamma_diffuse_basename)

df_gamma = read_hdf5(gamma_files, max_events=max_events)
df_proton = read_hdf5(proton_files, max_events=max_events)
df_gamma_diffuse = read_hdf5(gamma_diffuse_files, max_events=max_events)

removed_features =  ['alpha_err_init', 'intensity_err_init', 'length_err_init',
                     'phi_err_init', 'psi_err_init', 'r_err_init', 'width_err_init',
                     'x_err_init', 'y_err_init', 'intercept_err_init', 'slope_err_init',]

df_gamma = df_gamma.drop(removed_features, axis=1)
df_gamma_diffuse = df_gamma_diffuse.drop(removed_features, axis=1)
df_proton = df_proton.drop(removed_features, axis=1)
print(df_proton.columns)

target_features = ['particle']
train_features = ['alpha', 'alpha_err', 'intensity', 'intensity_err', 'kurtosis_l',
       'kurtosis_w', 'leakage', 'length', 'length_err', 'phi', 'phi_err',
       'psi', 'psi_err', 'r', 'r_err', 'skewness_l', 'skewness_w', 'width',
       'width_err', 'x', 'x_err', 'y', 'y_err', 'log_lh',  'intercept_err', 'slope', 'slope_err',
                  'wol', 'density',
       'density_l', 'density_w', 'area', 'log_intensity', 'particle']

def quality_cuts(df, intensity_cut=15):

    n_events = len(df)
    mask = (df['intensity'] >= intensity_cut) & (df['leakage'] <= 1.5) & (df['leakage'] >= 0.6) & (df['r'] <= 380)
    # mask = mask & (df['wol'] < 1) & (df['wol'] > 0.1)
    mask = np.arange(len(df))[mask]
    df = df.iloc[mask]
    with pd.option_context('mode.use_inf_as_null', True): # Drops infinite values
        df = df.dropna()

    n_dropped = n_events - len(df)
    print('N_events : {}\t N_dropped : {}\t N_kept : {}'.format(n_events, n_dropped, len(df)))
    return df


def plot_features(df_gamma, df_gamma_diffuse, df_proton, figure_path='figure_feature_distribution.pdf'):

    with PdfPages(figure_path) as pdf:

        features = df_gamma.columns

        for feature in features:

            if feature == 'valid':
                continue

            x, y, z = df_gamma[feature], df_proton[feature], df_gamma_diffuse[
                feature]
            x, y, z = x[np.isfinite(x)], y[np.isfinite(y)], z[np.isfinite(z)]

            if (len(x) == 0) or (len(y) == 0) or (len(z) == 0):
                continue


            bins = np.linspace(min(np.nanmin(x), np.nanmin(y), np.nanmin(z)),
                               max(np.nanmax(x), np.nanmax(y), np.nanmax(z)),
                               num=100)
            fig = plt.figure()
            axes = fig.add_subplot(111)
            hist_options = {'bins': bins, 'lw': 3,
                            'histtype': 'step', 'density': True}
            axes.hist(x, label='$\gamma$', **hist_options)
            axes.hist(y, label='Diffuse $p$', **hist_options)
            axes.hist(z, label='Diffuse $\gamma$', **hist_options)
            axes.legend(loc='best')
            axes.set_xlabel(feature)
            axes.set_yscale('log')
            axes.set_ylabel('Normalized counts []')
            pdf.savefig(fig)

plot_features(df_gamma, df_gamma_diffuse, df_proton, figure_path='test_before_cut.pdf')

df_gamma = quality_cuts(df_gamma, intensity_cut=size_cut)
df_gamma_diffuse = quality_cuts(df_gamma_diffuse, intensity_cut=size_cut)
df_proton = quality_cuts(df_proton, intensity_cut=size_cut)

plot_features(df_gamma, df_gamma_diffuse, df_proton, figure_path='test_after_cut.pdf')

df_proton_train, df_proton_test = train_test_split(df_proton, test_size=test_size)
df_gamma_diffuse_train, df_gamma_diffuse_test = train_test_split(df_gamma_diffuse, test_size=test_size)
df_gamma_train, df_gamma_test = train_test_split(df_gamma, test_size=test_size)

df_test = combine_datasets(df_proton_test, df_gamma_test)
df_train = combine_datasets(df_proton_train, df_gamma_diffuse_train)


df_train_X, df_train_y = df_train[train_features].drop(target_features, axis=1), df_train[target_features]
df_test_X, df_test_y = df_test[train_features].drop(target_features, axis=1), df_test[target_features]

features = df_train_X.columns

print("Test sample contains {:d} events\n".format(len(df_test)))
print("Train sample contains {:d} events\n".format(len(df_train)))

# rf_classifier = RandomForestClassifier(n_estimators=n_estimators, n_jobs=-1, verbose=3)

#  = 10
# max_depth = np.linspace(1, 32, num=n, endpoint=True).astype(int).tolist()
# max_depth.append(None)
max_depth = [None]
# n_estimators = np.linspace(50, 550, num=n, endpoint=True).astype(int).tolist()
n_estimators = [300]
cv = 2
parameters = {'n_estimators': n_estimators, 'max_depth': max_depth}

random_grid = GridSearchCV(RandomForestClassifier(n_jobs=-1, verbose=3), parameters, cv=cv)
random_grid.fit(df_train_X, df_train_y)
rf_classifier = random_grid.best_estimator_

prediction = rf_classifier.predict(df_test_X)
prob = rf_classifier.predict_proba(df_test_X)
prob_diffuse_gamma = rf_classifier.predict_proba(df_gamma_diffuse_test[train_features].drop(target_features, axis=1))
accuracy = accuracy_score(df_test_y, prediction)
roc_score_test = roc_auc_score(df_test_y, prediction)
roc_score_train = roc_auc_score(df_train_y, rf_classifier.predict(df_train_X))
importance = rf_classifier.feature_importances_
conf_matrix = confusion_matrix(df_test_y, prediction)
recall = recall_score(df_test_y, prediction)
best_params = random_grid.best_params_
print(random_grid.cv_results_)
print("Accuracy {:.6f}\nROC AUC score {:.6f}\nConfusion matrix {}\nRecall {:.6f}".format(accuracy, roc_score_test, conf_matrix, recall))
print("Best parameters : {}".format(best_params))
print("Train ROC AUC score {:.6f}".format(roc_score_train))
print("Test ROC AUC score {:.6f}".format(roc_score_test))

with PdfPages(os.path.join(output_dir, 'figure_classifier.pdf')) as pdf:

    axes = plot_feature_importance(features, importance)
    pdf.savefig(axes.get_figure())

    axes = plot_confusion_matrix(conf_matrix, [r'$\gamma$', r'$p$'])
    pdf.savefig(axes.get_figure())

    mask_proton = df_test_y == 1
    mask_proton = np.squeeze(mask_proton)
    print(prob.shape, mask_proton.shape)
    bins = np.linspace(0, 1, num=50)
    lw = 3
    fig = plt.figure()
    plt.hist(prob[mask_proton, 0], bins=bins, label='Diffuse $p$', histtype='step', density=True, lw=lw)
    plt.hist(prob[~mask_proton, 0], bins=bins, label='$\gamma$', histtype='step', density=True, lw=lw)
    plt.hist(prob_diffuse_gamma[:, 0], bins=bins, label='Diffuse $\gamma$', histtype='step', density=True, lw=lw)
    plt.ylabel('Density probability')
    plt.xlabel('Gamma-ness [a.u.]')
    plt.legend()
    pdf.savefig(fig)

    x, y, _ = roc_curve(df_test_y, prob[:, 1])
    fig = plt.figure(figsize=(10, 8))
    plt.plot(x, y, label='ROC AUC score : {:.4f}'.format(roc_score_test))
    plt.plot([0, 1], [0, 1], label='Coin flip', linestyle='--', color='k')
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.legend(loc='best')
    pdf.savefig(fig)

    fig = plt.figure()
    axes = fig.add_subplot(111)
    bins = np.linspace(0, np.pi / 2, num=100)
    resolution = df_gamma_test['alpha'].quantile(0.68)
    df_gamma_test.hist('alpha', bins=bins, ax=axes, histtype='step',
                  label='$\gamma$', density=True, lw=3)
    df_proton_test.hist('alpha', bins=bins, ax=axes, histtype='step',
                   label='Diffuse $p$', density=True, lw=3)
    axes.axvline(resolution,
                 label=r'$\alpha_{68}$ = %.2f [deg]' % np.degrees(resolution),
                 linestyle='--')
    axes.legend(loc='best')
    axes.set_yscale('log')
    axes.set_xlabel(r'$\alpha$ [rad]')
    axes.set_ylabel('probability density [rad$^{-1}$]')
    axes.set_title('Test data-set')
    pdf.savefig(fig)

    fig = plt.figure()
    axes = fig.add_subplot(111)
    bins = np.linspace(0, np.pi / 2, num=30)
    df_proton_train.hist('alpha', bins=bins, ax=axes, histtype='step',
                   label='Diffuse $p$', density=True, lw=3)
    df_gamma_diffuse_train.hist('alpha', bins=bins, ax=axes, histtype='step',
                          label='Diffuse $\gamma$', density=True, lw=3)
    axes.legend(loc='best')
    axes.set_yscale('log')
    axes.set_xlabel(r'$\alpha$ [rad]')
    axes.set_ylabel('probability density [rad$^{-1}$]')
    axes.set_title('Train data-set')
    pdf.savefig(fig)

    fig = plt.figure()
    axes = fig.add_subplot(111)
    bins = np.linspace(0, np.pi / 2, num=100)
    resolution = df_gamma['alpha'].quantile(0.68)
    df_gamma.hist('alpha', bins=bins, ax=axes, histtype='step',
                  label='$\gamma$', density=True, lw=3)
    df_proton.hist('alpha', bins=bins, ax=axes, histtype='step',
                         label='Diffuse $p$', density=True, lw=3)
    df_gamma_diffuse.hist('alpha', bins=bins, ax=axes, histtype='step',
                                label='Diffuse $\gamma$', density=True, lw=3)
    axes.axvline(resolution,
                 label=r'$\alpha_{68}$ = %.2f [deg]' % np.degrees(resolution),
                 linestyle='--')
    axes.legend(loc='best')
    axes.set_yscale('log')
    axes.set_xlabel(r'$\alpha$ [rad]')
    axes.set_ylabel('probability density [rad$^{-1}$]')
    axes.set_title('All data')
    pdf.savefig(fig)

    bins_intensity = np.linspace(np.log10(df_test_X['intensity'].min()), np.log10(df_test_X['intensity'].max()), num=20)
    roc_auc = np.zeros(len(bins_intensity)-1) * np.nan
    for i in range(len(bins_intensity)-1):

        mask = (np.log10(df_test_X['intensity']) > bins_intensity[i]) & (np.log10(df_test_X['intensity']) <= bins_intensity[i+1])

        if mask.sum() <= 100:
            continue

        mask = np.arange(len(df_test_X))[mask]
        df_X = df_test_X.iloc[mask]
        df_y = df_test_y.iloc[mask]

        pred = rf_classifier.predict(df_X)
        roc_auc[i] = roc_auc_score(df_y, pred)

    bins_width = np.diff(bins_intensity)
    bins_centers = bins_intensity[:-1] + 0.5 * bins_width
    fig = plt.figure()
    axes = fig.add_subplot(111)
    axes.errorbar(bins_centers, roc_auc, xerr=0.5 * bins_width, marker='o', color='k', linestyle='None')
    axes.set_xlabel(r'$\log_{10}\left( \frac{size}{{\rm p.e.}} \right)$')
    axes.set_ylabel('ROC AUC []')
    axes.set_ylim((0, 1))
    pdf.savefig(fig)

    bins_energy = np.linspace(np.log10(df_test['true_energy'].min()), np.log10(df_test['true_energy'].max()), num=20)
    roc_auc = np.zeros(len(bins_energy)-1) * np.nan
    for i in range(len(bins_energy)-1):

        mask = (np.log10(df_test['true_energy']) >= bins_energy[i]) * (np.log10(df_test['true_energy']) < bins_energy[i+1])

        if mask.sum() <= 100:
            continue

        mask = np.arange(len(df_test_X))[mask]
        df_X = df_test_X.iloc[mask]
        df_y = df_test_y.iloc[mask]

        pred = rf_classifier.predict(df_X)
        try:
            roc_auc[i] = roc_auc_score(df_y, pred)
        except ValueError:
            pass

    fig = plt.figure()
    axes = fig.add_subplot(111)
    bins_width = np.diff(bins_energy)
    bins_centers = bins_energy[:-1] + 0.5 * bins_width
    axes.errorbar(bins_centers, roc_auc, xerr=0.5 * bins_width, marker='o', color='k', linestyle='None')
    axes.set_xlabel(r'$\log_{10}\left( \frac{E}{{\rm TeV}} \right)$')
    axes.set_ylabel('ROC AUC []')
    axes.set_ylim((0, 1))
    pdf.savefig(fig)

output_filename = os.path.join(output_dir, 'classifier.pk')

# with open(output_filename, 'wb') as f:
#     pk.dump(rf_classifier, f)
