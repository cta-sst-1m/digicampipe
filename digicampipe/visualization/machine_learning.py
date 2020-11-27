import matplotlib.pyplot as plt
import numpy as np
import itertools

from scipy.optimize import curve_fit
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score

FOCAL = 5600
LABELS = {'alpha': r'$\alpha$ [rad]',
          'intensity': 'size [p.e.]',
          'kurtosis_l': 'longitudinal kurtosis []',
          'kurtosis_w': 'lateral kurtosis []',
          'leakage': 'containment []',
          'length': 'length [mm]',
          'phi': r'$\phi$ [rad]',
          'psi': r'$\psi$ [rad]',
          'r': '$r_{CM}$ [mm]',
          'skewness_l': 'longitudinal skewness []',
          'skewness_w': 'lateral skewness []',
          'width': 'width [mm]',
          'x': '$x_{CM}$ [mm]',
          'y': '$y_{CM}$ [mm]',
          'log_lh': r'$\ln \mathcal{L}$ []',
          'slope': r'$v^{-1}$ [ns/mm]',
          'velocity': r'$v$ [mm/ns]',
          'intercept': r'$t_{CM}$ [ns]',
          'wol': 'width over length []',
          'density': r'$\rho$ [p.e./mm$^2$]',
          'density_l': r'$\rho_l$ [p.e./mm]',
          'density_w': r'$\rho_w$ [p.e./mm]',
          'area': 'area [mm$^2$]',
          'log_intensity': r'$\log_{10}\left( \frac{{\rm size}}{{\rm p.e.}} \right)$',
          'true_energy': r'True energy [TeV]',
          'reco_energy': r'Reco energy [TeV]',
          'log_true_energy': r'$\log_{10}\left( \frac{{\rm True Energy}}{{\rm TeV}} \right)$',
          'log_energy': r'$\log_{10}\left( \frac{{\rm E}}{{\rm TeV}} \right)$',
          'log_reco_energy': r'$\log_{10}\left( \frac{{\rm Reco Energy}}{{\rm TeV}} \right)$',
          'disp_x': r'$DISP_x$ [mm]',
          'disp_y': r'$DISP_y$ [mm]',
          'disp_r': r'$DISP_r$ [mm]',
          'disp_theta': r'$DISP_{\theta}$ [rad]',
          'source_x': r'$x_{source}$ [mm]',
          'source_y': r'$y_{source}$ [mm]',
          'impact': 'Impact distance [m]',
          'angular_distance': r'$\delta$ [rad]',
          't_68': r'$t_{68}$ [ns]',
          'theta_2': r'$\theta^2$ [rad$^2$]',
          'alt': 'Alz [rad]',
          'az': 'Az [rad]',
          'core_x': r'$x_{core}$ [m]',
          'core_y': r'$y_{core}$ [m]',
          'event_id': 'Event ID',
          'h_first': r'$h_{first}$ [m]',
          'particle': 'Particle ID',
          'tel_alt': 'Tel Alt [rad]',
          'tel_az': 'Tel Az [rad]',
          'tel_id': 'Telescope ID',
          'x_max': r'$X_{max}$ [g/cm$^2$]',
          'valid': 'Valid',
          'delta': r'$\delta$ [rad]',
          'weights': 'Weights [Hz]'
          }
LABELS_WITHOUT_UNITS = {'width': r'$\sigma_w$',
                        'length': r'$\sigma_l$',
                        'area': 'area',
                        'kurtosis_l': 'long. kurtosis',
                        'kurtosis_w': 'lat. kurtosis',
                        'density': r'$\rho$',
                        'density_l': r'$\rho_l$',
                        'density_w': r'$\rho_w$',
                        'intensity': r'$\mu$',
                        't_68': r'$t_{68}$',
                        'skewness_l': 'long. skewness',
                        'psi': r'$\psi$',
                        'slope': r'$v^{-1}$',
                        'r': '$r_{CM}$',
                        'x': '$x_{CM}$',
                        'y': '$y_{CM}$',
                        'phi': '$\phi$',
                        'leakage': 'leakage',
                        }

def compute_bin_centers(bins):
    bins_errors = np.diff(bins)
    bins_centers = bins[:-1] + 0.5 * bins_errors

    return bins_centers, bins_errors


def plot_classifier_distribution(y_score, y_true):
    mask_proton = (y_true == 1)
    mask_proton = np.squeeze(mask_proton)
    bins = np.linspace(0, 1, num=50)
    lw = 3
    fig = plt.figure()
    axes = fig.add_subplot(111)
    axes.hist(y_score[mask_proton, 0], bins=bins, label='$p$', histtype='step',
              density=True, lw=lw)
    axes.hist(y_score[~mask_proton, 0], bins=bins, label='$\gamma$',
              histtype='step', density=True, lw=lw)
    # plt.hist(prob_diffuse_gamma[:, 0], bins=bins, label='Diffuse $\gamma$', histtype='step', density=True, lw=lw)
    axes.set_ylabel('Density probability')
    axes.set_xlabel('Gamma-ness [a.u.]')
    axes.legend(loc='best')

    return axes


def plot_feature_importance(features, importance, axes=None):

    if axes is None:

        fig = plt.figure()
        axes = fig.add_subplot(111)
    sort = np.argsort(importance)
    axes.barh(np.arange(len(features)), importance[sort], height=0.9)
    axes.axvline(1/len(features), label=r'$\frac{1}{N_{features}}$', linestyle='--', color='k')
    axes.set_yticks(np.arange(len(features)))
    labels = [LABELS_WITHOUT_UNITS[key] for key in features[sort]]
    axes.set_yticklabels(labels)
    axes.set_xlabel('Feature importance')
    axes.legend(loc='best')
    return axes


def plot_confusion_matrix(y_true, y_score, classes,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues,
                          axes=None):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """

    if axes is None:
        fig = plt.figure(figsize=(12, 8))
        axes = fig.add_subplot(111)

    y_score = np.argmax(y_score, axis=-1)
    cm = confusion_matrix(y_true, y_score)

    axes.imshow(cm, interpolation='nearest', cmap=cmap)
    axes.set_title(title)
    # axes.colorbar(label='count []')
    tick_marks = np.arange(len(classes))
    axes.set_xticks(tick_marks)
    axes.set_xticklabels(classes, rotation=45)
    axes.set_yticks(tick_marks)
    axes.set_yticklabels(classes)

    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100.

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        axes.text(j, i, '{:.2f} %'.format(cm[i, j]),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    # axes.tight_layout()
    axes.set_ylabel('True label')
    axes.set_xlabel('Predicted label')

    return axes


def plot_roc_curve(y_score, y_true, axes=None, label='', **kwargs):

    x, y, _ = roc_curve(y_true, y_score[:, 1])
    roc_auc = roc_auc_score(y_true, y_score[:, 1])
    if axes is None:
        fig = plt.figure(figsize=(10, 8))
        axes = fig.add_subplot(111)
    label = label + 'ROC AUC score : {:.4f}'.format(roc_auc)
    axes.step(x, y, label=label)
    axes.plot([0, 1], [0, 1], label='Coin flip', linestyle='--', color='k')
    axes.set_xlabel('False positive rate')
    axes.set_ylabel('True positive rate')
    axes.legend(loc='best')

    return axes


def plot_roc_curve_energy(X, y_true, y_score, bins=None):

    x = np.log10(X['true_energy'])

    bins_energy = np.linspace(-1, 3, num=21)
    bins_err = 0.5 * np.diff(bins_energy)
    bins_centers = bins_energy[:-1] + bins_err
    roc_auc = np.zeros(len(bins_energy) - 1) * np.nan
    for i in range(len(bins_energy) - 1):

        mask = (x >= bins_energy[i]) * (x < bins_energy[i + 1])

        if mask.sum() <= 100:
            continue

        try:
            roc_auc[i] = roc_auc_score(y_true[mask], y_score[mask][:, 1])
        except ValueError:
            pass

    fig = plt.figure()
    axes = fig.add_subplot(111)
    axes.errorbar(bins_centers, roc_auc, xerr=bins_err, marker='o',
                  color='k', linestyle='None')
    axes.set_xlabel(r'True $\log_{10}\left( \frac{E}{{\rm TeV}} \right)$')
    axes.set_ylabel('ROC AUC []')
    axes.set_ylim((0, 1))

    return axes


def plot_2d_source(df, bins=100):

    fig = plt.figure()
    axes = fig.add_subplot(111)
    H = axes.hist2d(df['source_x'],
                df['source_y'],
                bins=bins, norm=LogNorm())
    axes.set_xlabel('$x_{source}$ [mm]')
    axes.set_ylabel('$y_{source}$ [mm]')
    axes.legend(loc='best')
    fig.colorbar(H[3], label='count []')
    return axes


def plot_2d_disp(df, bins=100):
    fig = plt.figure()
    axes = fig.add_subplot(111)
    H = axes.hist2d(df['disp_x'],
                df['disp_y'],
                bins=bins, norm=LogNorm())
    axes.set_xlabel('$DISP_x$ [mm]')
    axes.set_ylabel('$DISP_y$ [mm]')
    axes.legend(loc='best')
    fig.colorbar(H[3], label='count []')
    return axes


def plot_delta_disp(y_true, y_fit, bins=(100, 100)):

    dx, dy = y_true['disp_x'] - y_fit[:, 0], y_true['disp_y'] - y_fit[:, 1]
    mask = np.isfinite(dx) * np.isfinite(dy)
    dx = dx[mask]
    dy = dy[mask]

    mean_x = np.mean(dx)
    mean_y = np.mean(dy)
    dr = np.sqrt((dx - mean_x) ** 2 + (dy - mean_y) ** 2)
    r_68 = np.quantile(dr, 0.68)

    x_bins = np.linspace(dx.min(), dx.max(), num=bins[0])
    y_bins = np.linspace(dy.min(), dy.max(), num=bins[1])

    label = '$R_{68} =$ ' + '{:.2f} [mm]\n'.format(r_68) + \
         '$(x, y) =$' + '({:.4f}, {:.4f}) [mm]'.format(mean_x, mean_y)

    fig = plt.figure()
    axes = fig.add_subplot(111)

    theta = np.linspace(0, np.pi * 2, num=1000)
    H = axes.hist2d(dx, dy, bins=[x_bins, y_bins], norm=LogNorm(),)
    axes.plot(mean_x + r_68 * np.cos(theta), mean_y + r_68 * np.sin(theta), label=label, color='k', linestyle='--')
    axes.set_xlabel(r'$\Delta DISP_{x}$ [mm]')
    axes.set_ylabel(r'$\Delta DISP_{y}$ [mm]')
    axes.set_xlim((-500, 500))
    axes.set_ylim((-500, 500))
    axes.legend(loc='best')
    fig.colorbar(H[3], label='count []')

    return axes


def plot_r68(y_true, y_score, energy, kind='true', bins=15):

    if kind == 'true':
        x_label = r'True $\log_{10}\left(\frac{E}{{\rm TeV}}\right)$'

    elif kind == 'reco':
        x_label = r'Reco $\log_{10}\left(\frac{E}{{\rm TeV}}\right)$'

    x = energy
    x = np.log10(x)
    x_energy = np.linspace(-1, 3, num=21)
    x_err = np.diff(x_energy) * 0.5
    x_centers = x_err + x_energy[:-1]
    r_68 = np.zeros(len(x_energy) - 1) * np.nan

    dx, dy = y_true['disp_x'] - y_score[:, 0], y_true['disp_y'] - y_score[:, 1]
    mask = np.isfinite(dx) * np.isfinite(dy)
    dx = dx[mask]
    dy = dy[mask]

    mean_x = np.mean(dx)
    mean_y = np.mean(dy)
    dr = np.sqrt((dx - mean_x) ** 2 + (dy - mean_y) ** 2)

    for i in range(len(x_energy) - 1):
        mask = (x > x_energy[i]) * (x <= x_energy[i + 1])

        if mask.sum() < 100:
            continue

        r_68[i] = np.quantile(dr[mask], 0.68)

    r_68 = r_68 / FOCAL
    fig = plt.figure()
    axes = fig.add_subplot(111)
    axes.errorbar(x_centers, np.degrees(r_68), xerr=x_err, label='$R_{68}$', marker='o', color='k', linestyle='None')
    axes.set_xlabel(x_label)
    axes.set_ylabel('[deg]')
    axes.legend(loc='best')
    # axes.set_ylim((-0.3, 0.3))
    return axes


def plot_delta_energy(y_true, y_score, kind='true', bins=(100, 100)):

    if kind == 'true':
        x = y_true
        x_label = r'True $\log_{10}\left(\frac{E}{{\rm TeV}}\right)$'

    elif kind == 'reco':
        x = y_score
        x_label = r'Reco $\log_{10}\left(\frac{E}{{\rm TeV}}\right)$'

    y = (y_score - y_true) / y_true

    mask = np.isfinite(x) * np.isfinite(y)
    x, y = x[mask], y[mask]
    x = np.log10(x)
    x_bins = np.linspace(x.min(), x.max(), num=bins[0])
    y_bins = np.linspace(-2, 2, num=bins[1])

    fig = plt.figure()
    axes = fig.add_subplot(111)
    H = axes.hist2d(x, y, bins=[x_bins, y_bins], norm=LogNorm(), )
    axes.set_xlabel(x_label)
    axes.set_ylabel(r'$\frac{E_{reco} - E_{true}}{E_{true}}$ []')
    axes.set_xlim(0, 3)
    axes.legend(loc='best')
    fig.colorbar(H[3], label='count []')
    return axes


def plot_resolution_energy(y_true, y_score, kind='true'):

    if kind == 'true':
        x = np.log10(y_true['true_energy'])
        x_label = r'True $\log_{10}\left(\frac{E}{{\rm TeV}}\right)$'

    elif kind == 'reco':
        x = np.log10(y_score)
        x_label = r'Reco $\log_{10}\left(\frac{E}{{\rm TeV}}\right)$'
    x_bins = np.linspace(-1, 3, num=20+1)
    x_err = 0.5 * np.diff(x_bins)
    x_centers = x_bins[:-1] + x_err

    diff_reco = np.zeros(len(x_bins) - 1) * np.nan
    diff_reco_median = np.zeros(len(x_bins) - 1) * np.nan
    diff_68_reco = np.zeros(len(x_bins) - 1) * np.nan
    diff_reco_std = np.zeros(len(x_bins) - 1) * np.nan
    diffs = (y_score - y_true['true_energy']) / y_true['true_energy']

    for i in range(len(x_bins) - 1):
        mask = (x <= x_bins[i + 1]) * (x > x_bins[i])

        if mask.sum() > 100:
            diff = diffs[mask]
            diff_reco[i] = np.mean(diff)
            diff_reco_median[i] = np.median(diff)
            diff_68_reco[i] = np.quantile(np.abs(diff), 0.68)
            diff_reco_std[i] = np.std(diff)

    fig = plt.figure()
    axes = fig.add_subplot(111)
    axes.errorbar(x_centers, diff_68_reco, xerr=x_err, color='r', linestyle='None', marker='o', label='68.2 %')
    axes.errorbar(x_centers, diff_reco_std, xerr=x_err, color='g', linestyle='None', marker='o', label='Std')
    axes.errorbar(x_centers, diff_reco, xerr=x_err, color='k', linestyle='None', marker='o', label='Mean')
    axes.errorbar(x_centers, diff_reco_median, xerr=x_err, color='b', linestyle='None', marker='o', label='Median')
    axes.set_xlabel(x_label)
    axes.set_ylabel(r'$\frac{E_{reco} - E_{true}}{E_{true}}$ []')
    axes.set_xlim(0, 3)
    axes.set_ylim(-0.4, 1)
    axes.legend(loc='best')

    return axes


def plot_features(feature, df_gamma, df_gamma_diffuse, df_proton):

    if not (feature in LABELS.keys()):
        raise KeyError
    x, y, z = df_gamma[feature], df_proton[feature], df_gamma_diffuse[
        feature]
    x, y, z = x[np.isfinite(x)], y[np.isfinite(y)], z[np.isfinite(z)]

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
    axes.set_xlabel(LABELS[feature])
    axes.set_yscale('log')
    axes.set_ylabel('Normalized counts []')

    return axes


def plot_weighted_features(feature, df_gamma, df_proton, n_bins=100):

    if not (feature in LABELS.keys()):
        raise KeyError
    x, y = df_gamma[feature], df_proton[feature]
    x, y = x[np.isfinite(x)], y[np.isfinite(y)]

    bins = np.linspace(min(np.nanmin(x), np.nanmin(y)),
                       max(np.nanmax(x), np.nanmax(y)),
                       num=n_bins)
    fig = plt.figure()
    axes = fig.add_subplot(111)
    hist_options = {'bins': bins, 'lw': 3,
                    'histtype': 'step',}
    axes.hist(x, label='$\gamma$', weights=df_gamma['weights'], **hist_options,)
    axes.hist(y, label='Diffuse $p$', weights=df_proton['weights'], **hist_options)
    axes.legend(loc='best')
    axes.set_xlabel(LABELS[feature])
    axes.set_yscale('log')
    axes.set_ylabel('Counts [Hz]')

    return axes


def plot_init_vs_fit(df, feature, bins=100):
    x = np.array(df[feature])
    y = np.array(df[feature + '_init'])
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]

    if feature == 'intensity':
        x, y = np.log10(x), np.log10(y)
        feature = 'log_intensity'

    bins = np.linspace(min(x.min(), y.min()), max(x.max(), y.max()), num=bins)
    fig = plt.figure()
    axes = fig.add_subplot(111)
    H = axes.hist2d(x, y, bins=[bins, bins], norm=LogNorm())
    axes.plot(bins, bins, linestyle='--', color='w')
    axes.set_xlabel('Fit ' + LABELS[feature])
    axes.set_ylabel('Initial ' + LABELS[feature])
    fig.colorbar(H[3], label='count []', ax=axes)
    return axes

def plot_2d_histogram(df, x_feature, y_feature, bins=100, log=(False, False), limits=((-np.inf, np.inf), (-np.inf, np.inf))):

    x = np.array(df[x_feature])
    y = np.array(df[y_feature])

    if log[0]:
        x = np.log10(x)
        x_feature = 'log_' + x_feature

    if log[1]:
        y = np.log10(y)
        y_feature = 'log_' + y_feature

    if isinstance(bins, int) or isinstance(bins, float):
        bins = (bins, bins)

    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]

    bins_x = np.linspace(max(x.min(), limits[0][0]), min(x.max(), limits[0][1]), num=bins[0])
    bins_y = np.linspace(max(y.min(), limits[1][0]), min(y.max(), limits[1][1]), num=bins[1])
    fig = plt.figure()
    axes = fig.add_subplot(111)
    H = axes.hist2d(x, y, bins=[bins_x, bins_y], norm=LogNorm())
    axes.set_xlabel(LABELS[x_feature])
    axes.set_ylabel(LABELS[y_feature])
    fig.colorbar(H[3], label='count []', ax=axes)
    return axes


def plot_grid_cv(cv_results):

    params = cv_results['params']
    param_x = []
    for i in range(len(params)):

        param_x.append(params[i]['n_estimators'])

    fig = plt.figure()
    axes = fig.add_subplot(111)
    try:
        axes.errorbar(param_x, cv_results['mean_train_score'], yerr=cv_results['std_train_score'], marker='o', linestyle='None',  label='Train')
    except KeyError:
        pass
    axes.errorbar(param_x, cv_results['mean_test_score'], yerr=cv_results['std_test_score'], marker='o', linestyle='None', label='Test')

    axes.set_xlabel('Number of trees')
    axes.set_ylabel('Score []')
    axes.legend(loc='best')

    return axes




def plot_grid_cv_2(grid_cv, params_x, params_y):

    cv_results = grid_cv.cv_results_
    best_index = grid_cv.best_index_
    # y_label = grid_cv.scorer_
    y_label = 'Score []'
    sub_title = 'Number of cross-validation : {}'.format(grid_cv.n_splits_)

    params = cv_results['params']
    keys = list(params[0].keys())
    n_params = len(keys)
    assert n_params == 2

    train_score = cv_results['mean_train_score']
    std_train_score = cv_results['std_train_score']
    test_score = cv_results['mean_test_score']
    std_test_score = cv_results['std_test_score']
    best_mean_test = cv_results['mean_test_score'][best_index]
    best_std_test = cv_results['std_test_score'][best_index]
    train_score = train_score.reshape((len(params_y), len(params_x))).T
    std_train_score = std_train_score.reshape((len(params_y), len(params_x))).T
    test_score = test_score.reshape((len(params_y), len(params_x))).T
    std_test_score = std_test_score.reshape((len(params_y), len(params_x))).T

    fig = plt.figure()
    axes = fig.add_subplot(111)

    for j in range(len(params_y)):

        label = 'Train Maximum Depth : {:.1f}'.format(params_y[j])
        options = {'linestyle': '--', 'marker': 'o'}
        p = axes.errorbar(params_x, train_score[:, j], yerr=std_train_score[:, j], label=label, **options)
        label = 'Test Maximum Depth : {:.1f}'.format(params_y[j])
        color = p[0].get_color()
        options = {'linestyle': '-', 'color': color, 'marker': 'o'}
        axes.errorbar(params_x, test_score[:, j], yerr=std_test_score[:, j], label=label, **options)

    options = {'linestyle': 'None', 'color': 'k', 'marker': 'x'}
    label = 'Best Maximum depth : {:.1f}'.format(grid_cv.best_params_['max_depth'])
    axes.errorbar(grid_cv.best_params_['n_estimators'], best_mean_test, yerr=best_std_test, label=label, **options)
    axes.set_xlabel('Number of trees')
    axes.set_ylabel(y_label)
    axes.set_yscale('log')
    axes.legend(loc='best')

    return axes


def plot_impact_parameter(df, bins=200):

    fig = plt.figure()
    axes = fig.add_subplot(111)
    H = axes.hist2d(df['core_x'], df['core_y'], norm=LogNorm(), bins=bins)
    axes.set_xlabel('$x$ [m]')
    axes.set_ylabel('$y$ [m]')
    fig.colorbar(H[3], label='count []', ax=axes)

    return axes


def plot_impact_parameter_1d(count, bins_impact, **kwargs):

    fig = plt.figure()
    axes = fig.add_subplot(111)
    data = count.sum(axis=(2, 0))
    bins_x, bins_err = compute_bin_centers(bins_impact)
    axes.plot(bins_x, data, **kwargs)
    axes.set_xlabel(LABELS['impact'])
    axes.set_ylabel('count []')

    return axes


def plot_impact_parameter_vs_energy(df, bins=200):

    fig = plt.figure()
    axes = fig.add_subplot(111)
    r = np.sqrt(df['core_x']**2 + df['core_y']**2)
    e = np.log10(df['true_energy'])
    H = axes.hist2d(e, r, norm=LogNorm(), bins=bins)
    axes.set_xlabel(r'$\log_{10} \left( \frac{{\rm Energy}}{{\rm TeV}} \right)$')
    axes.set_ylabel('$r$ [m]')
    fig.colorbar(H[3], label='count []', ax=axes)

    return axes


def plot_3d_histo(counts_1, bins, counts_2=None, axis=-1, label='count []', figsize=(10, 8), vmin=None, vmax=None, **kwargs):

    bins = bins.copy()
    bins.pop(axis)
    X, Y = bins[0], bins[1]
    #X = X[:-1] + 0.5 * np.diff(X)
    #Y = Y[:-1] + 0.5 * np.diff(Y)
    # X, Y = np.meshgrid(X, Y)

    if len(counts_1.shape) > 2:
        if counts_2 is None:
            mask = np.isfinite(counts_1)
            counts_1[~mask] = 0
            data = counts_1.sum(axis=axis)

        else:

            mask = np.isfinite(counts_1)
            counts_1[~mask] = 0
            data_1 = counts_1.sum(axis=axis)

            mask = np.isfinite(counts_1)
            counts_2[~mask] = 0
            data_2 = counts_2.sum(axis=axis)
            data = data_1 / data_2
    else:

        data = counts_1

    fig = plt.figure(figsize=figsize)
    axes = fig.add_subplot(111)
    im = axes.pcolor(X, Y, data.T, norm=LogNorm(vmin=vmin, vmax=vmax), **kwargs)
    fig.colorbar(im, label=label, ax=axes)

    return axes


def plot_trigger_probability(df, bins=(200, 400)):

    fig = plt.figure()
    axes = fig.add_subplot(111)
    r = np.sqrt(df['core_x']**2 + df['core_y']**2)
    e = np.log10(df['true_energy'])

    bins_x = np.linspace(e.min(), e.max(), num=bins[0])
    bins_x_centers = np.diff(bins_x) * 0.5 + bins_x[:-1]
    bins_y = np.linspace(0, r.max(), num=bins[1])
    bins_y_centers = np.diff(bins_y) * 0.5 + bins_y[:-1]
    count, _, _ = np.histogram2d(e, r, bins=[bins_x, bins_y])
    n_events = count.sum(axis=1)
    count = count.T / n_events
    X, Y = np.meshgrid(bins_x_centers, bins_y_centers)
    im = axes.pcolormesh(X, Y, count, norm=LogNorm())
    axes.set_xlabel(r'$\log_{10} \left( \frac{{\rm Energy}}{{\rm TeV}} \right)$')
    axes.set_ylabel('$r$ [m]')
    fig.colorbar(im, label='Probability []', ax=axes)

    return axes

def compute_effective_area(counts_events, counts_trigger, bins_energy, bins_impact, bins_theta=None):


    prob = counts_trigger / counts_events
    mask = counts_events <=0
    prob[mask] = 0

    bins_x_centers, bin_x_errors = compute_bin_centers(bins_energy)
    bins_y_centers, bin_y_errors = compute_bin_centers(bins_impact)

    if len(prob.shape) == 2:

        a_eff = 2 * np.pi * np.sum(prob * bins_y_centers * bin_y_errors, axis=-1)

    elif len(prob.shape) == 3:

        if bins_theta is not None:

            bins_omega = 2 * np.pi * (1 - np.cos(bins_theta))
            bins_z_centers, bin_z_errors = compute_bin_centers(bins_omega)

        else:
            bin_z_errors = 1
        a_eff = 2 * np.pi * np.sum(np.sum(prob * bin_z_errors, axis=-1) * bins_y_centers * bin_y_errors, axis=-1)

    return a_eff, bins_x_centers, bin_x_errors


def crab_spectrum(energy, type='magic'):

    """
    gamma = 2.39
    energy_c = 14.3
    spectrum = 3.76 * 1E-11 * 1E4 * energy ** -gamma * np.exp(
        -energy / energy_c)
    """

    if type == 'magic':
        spectrum = 3.23 * 1E-11 * energy ** (-2.47 - 0.24 * np.log10(energy))  # TeV-1 cm-2 s-1
        spectrum = spectrum * 10000  # TeV-1 m-2 s-1

    else:
        const = 2.83 * 1E-7 # TeV^-1 m^-2 s^-1
        spectral_index_real = -2.62
        e0 = 1 # TeV
        spectrum = const * (energy / e0)**spectral_index_real

    return spectrum


def proton_spectrum(energy):

    gamma = 2.7
    spectrum = 0.096 * energy ** -gamma

    return spectrum


def get_effective_area_weights(counts_events, counts_trigger, bins, limit_energy, kind='proton'):

    bins_energy, bins_impact, bins_theta = bins
    bins_energy = 10 ** bins_energy
    a_eff, bins_x_centers, bin_x_errors = compute_effective_area(counts_events,
                                                                 counts_trigger,
                                                                 bins_energy,
                                                                 bins_impact,
                                                                 bins_theta)
    energy = bins_x_centers
    mask = (energy >= limit_energy[0]) * (energy <= limit_energy[1])
    x = energy[mask]
    y = a_eff[mask]
    popt = fit_effective_area(y, x)

    return popt


def compute_differential_trigger_rate(counts_events, counts_trigger, bins, limit_energy, kind='proton'):

    bins_energy, bins_impact, bins_theta = bins
    bins_energy = 10**bins_energy
    a_eff, bins_x_centers, bin_x_errors = compute_effective_area(counts_events, counts_trigger, bins_energy, bins_impact, bins_theta)
    energy = bins_x_centers

    mask = (energy >= limit_energy[0]) * (energy <= limit_energy[1])
    x = energy[mask]
    y = a_eff[mask]
    popt = fit_effective_area(y, x)
    a_eff = 10**(log_effective_area(energy, *popt))

    if kind == 'proton':
        gamma = 2.7
        spectrum = 0.096 * energy**-gamma

    elif kind == 'gamma' or kind == 'crab':

        spectrum = crab_spectrum(energy)
    diff_trig = spectrum * a_eff

    return diff_trig


def plot_differential_trigger_rate(counts_events, counts_trigger, bins_energy, bins_impact, bins_theta=None, axes=None, kind='proton', limit_energy=(-np.inf, np.inf), **kwargs):

    if axes is None:
        fig = plt.figure()
        axes = fig.add_subplot(111)

    bins = [bins_energy, bins_impact, bins_theta]
    bins_x_centers, bin_x_errors = compute_bin_centers(10**bins_energy)
    energy = bins_x_centers

    diff_rate = compute_differential_trigger_rate(counts_events, counts_trigger, bins=bins, limit_energy=limit_energy, kind=kind)
    energy_threshold = energy[np.argmax(diff_rate)]
    total_rate = np.trapz(diff_rate, energy)
    label = 'Total rate {:.2f} [mHz] \nEnergy threshold = {:.2f} TeV'.format(total_rate*1E3, energy_threshold)

    if 'label' in kwargs.keys():
        kwargs['label'] = kwargs['label'] + '\n' + label
    else:
        kwargs['label'] = label

    x, x_err = compute_bin_centers(bins_energy)
    axes.errorbar(x, diff_rate, xerr=x_err, **kwargs)
    axes.set_xlabel(LABELS['log_true_energy'])
    axes.set_ylabel('Differential trigger rate [Hz/TeV]')
    axes.set_yscale('log')

    return axes


def log_effective_area(energy, a, b, e_c, gamma):

    log_a_eff = a + b * np.log10(energy) - np.log10(1 + (energy/e_c)**(-gamma))

    return log_a_eff


def fit_effective_area(a_eff, energy):

    mask = np.isfinite(a_eff) * (a_eff > 0)
    a_eff, energy = a_eff[mask], energy[mask]

    p0 = [4, 0.25, 10 ** 0.75, 2]
    popt, pcov = curve_fit(log_effective_area, energy,
                           np.log10(a_eff), p0=p0)

    return popt


def plot_effective_area(counts_events, counts_trigger, bins_energy, bins_impact, bins_theta=None, axes=None, limit_energy=(-np.inf, np.inf), **kwargs):

    if axes is None:
        fig = plt.figure()
        axes = fig.add_subplot(111)

    a_eff, bins_x_centers, bin_x_errors = compute_effective_area(counts_events, counts_trigger, bins_energy, bins_impact, bins_theta)

    e = 10**bins_x_centers
    mask = (e >= limit_energy[0]) * (e <= limit_energy[1])
    popt = fit_effective_area(a_eff[mask], e[mask])

    x_fit = np.linspace(bins_x_centers.min(), bins_x_centers.max(),
                        num=len(bins_x_centers) * 10)

    y_fit = 10**(log_effective_area(10**x_fit, *popt))
    if bins_theta is not None:

        axes.set_ylabel('Effective area [m$^2\cdot$ sr]')

    else:
        axes.set_ylabel('Effective area [m$^2$]')

    p = axes.plot(x_fit, y_fit, )
    color = p[-1].get_color()
    linestyle = kwargs.pop('linestyle', 'None')
    fit_label = "\n" + "$\log_{10}(A_{eff})=" + "{:.3g}".format(popt[0]) + \
    " + {:.3g}".format(popt[1]) + \
    "*\log_{10}(E) - \log_{10}(1 + (E/" + "{:.3g}".format(popt[2]) + \
    ")^{" + "{:.3g}".format(popt[3]) + "})$"
    label = kwargs.pop('label', '') + fit_label
    axes.errorbar(bins_x_centers, a_eff, xerr=bin_x_errors, linestyle=linestyle, color=color, label=label, **kwargs)
    axes.set_xlabel(LABELS['log_true_energy'])
    axes.set_yscale('log')

    return axes


def get_rate_gamma(bins_energy, bins_impact,):
    e, d_e = compute_bin_centers(10 ** bins_energy)
    r, d_r = compute_bin_centers(bins_impact)

    rate = np.zeros((len(e), len(r)))
    spectrum = crab_spectrum(energy=e)

    for i in range(len(r)):
            rate[:, i] = spectrum * d_e * (d_r[i]**2 + 2 * bins_impact[i] * d_r[i]) * np.pi

    return rate


def get_rate_proton(bins_energy, bins_impact, bins_theta):

    e, d_e = compute_bin_centers(10**bins_energy)
    r, d_r = compute_bin_centers(bins_impact)
    bins_omega = 2 * np.pi * (1 - np.cos(bins_theta))
    omega, d_omega = compute_bin_centers(bins_omega)

    rate = np.zeros((len(e), len(r), len(omega)))
    spectrum = proton_spectrum(energy=e)

    for i in range(len(r)):
        for j in range(len(omega)):

            rate[:, i, j] = spectrum * d_e * (d_r[i]**2 + 2 * bins_impact[i] * d_r[i]) * np.pi * d_omega[j]

    return rate