from os.path import join
import joblib
import pickle as pk
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import argparse
from digicampipe.visualization.machine_learning import crab_spectrum,LABELS, plot_roc_curve, plot_confusion_matrix, plot_grid_cv_2, LABELS_WITHOUT_UNITS, plot_feature_importance
from sklearn.metrics import roc_auc_score
from tqdm import tqdm_notebook as tqdm
from scipy.stats import beta as beta_stat
from astropy.io import fits
from scipy.stats.mstats import mjci
from scipy.optimize import brentq

def add_reconstructed_params(df, df_X, rf_classifier, rf_disp, rf_energy, focal, source_x=0, source_y=0, n_telescopes=1):

    gammaness = rf_classifier.predict_proba(df_X)[:, 0]
    disp = rf_disp.predict(df_X)
    energy = rf_energy.predict(df_X)
    df['gammaness'] = gammaness
    df['reco_disp_x'] = disp[:, 0]
    df['reco_disp_y'] = disp[:, 1]
    df['reco_source_x'] = df['reco_disp_x'] + df['x']
    df['reco_source_y'] = df['reco_disp_y'] + df['y']
    dr = np.sqrt((df['reco_source_x'] - source_x) ** 2 + (
                df['reco_source_y'] - source_y) ** 2)
    df['theta_2'] = np.arctan(dr / focal) ** 2
    df['reco_energy'] = energy
    df['weights'] /= n_telescopes

    return df


def sigma_lima(n_on, n_off, alpha):


    sigma_lima = np.sqrt(2 * (n_on * np.log((1 + alpha) / alpha * n_on / (n_on + n_off)) + n_off * np.log((1 + alpha) * (n_off / (n_on + n_off)))))

    return sigma_lima


def relative_sensitivity(
        n_on,
        n_off,
        alpha,
        target_significance=5,
        significance_function=sigma_lima,
):
    """
    Calculate the relative sensitivity defined as the flux
    relative to the reference source that is detectable with
    significance ``target_significance``.

    Given measured ``n_on`` and ``n_off``,
    we estimate the number of gamma events ``n_signal`` as ``n_on - alpha * n_off``.

    The number of background events ``n_background` is estimated as ``n_off * alpha``.

    In the end, we find the relative sensitivity as the scaling factor for ``n_signal``
    that yields a significance of ``target_significance``.

    The reference time should be incorporated by appropriately weighting the events
    before calculating ``n_on`` and ``n_off``.

    Parameters
    ----------
    n_on: int or array-like
        Number of signal-like events for the on observations
    n_off: int or array-like
        Number of signal-like events for the off observations
    alpha: float
        Scaling factor between on and off observations.
        1 / number of off regions for wobble observations.
    significance: float
        Significance necessary for a detection
    significance_function: function
        A function f(n_on, n_off, alpha) -> significance in sigma
        Used to calculate the significance, default is the Li&Ma
        likelihood ratio test formula.
        Li, T-P., and Y-Q. Ma.
        "Analysis methods for results in gamma-ray astronomy."
        The Astrophysical Journal 272 (1983): 317-324.
        Formula (17)
    """
    if np.isnan(n_on) or np.isnan(n_off):
        return np.inf

    if n_on < 0 or n_off < 0:
        return np.inf

    n_background = n_off * alpha
    n_signal = n_on - n_background

    if n_signal <= 0:
        return np.inf

    def equation(relative_flux):
        n_on = n_signal * relative_flux + n_background
        s = significance_function(n_on, n_off, alpha)
        return s - target_significance

    try:
        # brentq needs a lower and an upper bound
        # we will use the simple, analytically  solvable significance formula and scale it
        # with 10 to be sure it's above the Li and Ma solution
        # so rel * n_signal / sqrt(n_background) = target_significance
        if n_off > 1:
            relative_flux_naive = target_significance * np.sqrt(
                n_background) / n_signal
            upper_bound = 10 * relative_flux_naive
            lower_bound = 0.01 * relative_flux_naive
        else:
            upper_bound = 100
            lower_bound = 1e-4

        relative_flux, out = brentq(equation, lower_bound, upper_bound,
                                    full_output=True)

    except (RuntimeError, ValueError) as e:
        print(
            "Could not calculate relative significance for"
            f" n_signal={n_signal:.1f}, n_off={n_off:.1f}, returning nan {e}"
        )
        return np.inf

    if not out.converged:
        return np.inf

    return relative_flux


def compute_sigma(n_gamma, n_proton, t_on, t_off, n_g_err, n_p_err):
    alpha = t_on / t_off
    n_on = n_gamma + alpha * n_proton
    n_off = n_proton

    sigma_min = np.zeros(len(n_gamma))
    sigma_max = np.zeros(len(n_gamma))
    sigma = np.zeros(len(n_gamma))

    n_on_err = np.sqrt(n_g_err ** 2 + (alpha * n_p_err) ** 2)
    n_off_err = n_p_err

    for i in range(len(sigma)):
        sigma[i] = relative_sensitivity(n_on=n_on[i], n_off=n_off[i],
                                        alpha=alpha)
        sigma_min[i] = relative_sensitivity(n_on=n_on[i] - n_on_err[i],
                                            n_off=n_off[i] + n_off_err[i],
                                            alpha=alpha)
        sigma_max[i] = relative_sensitivity(n_on=n_on[i] + n_on_err[i],
                                            n_off=n_off[i] - n_off_err[i],
                                            alpha=alpha)
        # sigma_1[i] = relative_sensitivity(n_on=n_on[i] + n_on_err[i], n_off=n_off[i] + n_off_err[i], alpha=alpha)
        # sigma_2[i] = relative_sensitivity(n_on=n_on[i] - n_on_err[i], n_off=n_off[i] - n_off_err[i], alpha=alpha)
        # sigma_2[i] = relative_sensitivity(n_on=n_on[i], n_off=n_off[i] - n_off_err[i], alpha=alpha)

    sigma_err = (sigma_min - sigma_max) / 2

    mask = (n_gamma >= 10) & (
                n_gamma >= (0.05 * n_proton * alpha)) & (n_proton > 0)
    sigma[~mask] = np.inf
    sigma_err[~mask] = np.inf

    return sigma, sigma_err, sigma_min, sigma_max


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Process some value.')
    parser.add_argument('--output',
                        default=None,
                        type=str, help='Output directory')
    parser.add_argument('--input', default=None,
                        type=str, help='Input directory where dl1 and rf models are stored')
    parser.add_argument('--electron', default=False,
                        type=bool, help='Use electrons for sensitivity')
    parser.add_argument('--focal',
                        default=28000,
                        type=float, help='Focal length')
    parser.add_argument('--source_x',
                        default=0,
                        type=float, help='Source x position')
    parser.add_argument('--source_y',
                        default=0,
                        type=float, help='Source y position')
    parser.add_argument('--test_size', default=None,
                        type=float, help='Relative size of the proton test sample')

    parser.add_argument('--energy_min', default=-2,
                        type=int, help='log10 of minimum energy bin (TeV)')

    parser.add_argument('--energy_max', default=3,
                        type=int, help='log10 of maximum energy bin (TeV)')

    parser.add_argument('--n_telescope', default=None,
                        type=int, help='Number of telescopes simulated for mono reconstruction')

    parser.add_argument('--theta', default=None,
                        type=float,
                        help='Theta cut to apply in degrees')
    args = parser.parse_args()

    input_dir = args.input
    output_dir = args.output
    focal = args.focal
    source_x, source_y = args.source_x, args.source_y
    test_size = args.test_size

    n_bins = (args.energy_max - args.energy_min) * 5 + 1  # 5 bins per energy decade
    bins_energy_CTA = np.linspace(args.energy_min, args.energy_max, num=n_bins)
    n_telescopes = args.n_telescope

    bins_energy_CTA_width = np.diff(10**bins_energy_CTA)
    bins_energy_CTA_mid = 10**bins_energy_CTA[:-1] + 0.5 * bins_energy_CTA_width

    figsize = (10, 8)


    rf_classifier = joblib.load(join(input_dir, 'classifier.sav'))
    rf_energy = joblib.load(join(input_dir, 'energy_regressor.sav'))
    rf_disp = joblib.load(join(input_dir, 'disp_regressor.sav'))
    results_classifier = pk.load(open(join(input_dir, 'gridcv_classifier.pk'), 'rb'))
    results_energy = pk.load(open(join(input_dir, 'gridcv_energy_regressor.pk'), 'rb'))
    results_disp = pk.load(open(join(input_dir, 'gridcv_disp_regressor.pk'), 'rb'))

    df_test_classifier = pd.read_hdf(join(input_dir, 'classifier_test.h5'),
                                     'data')
    df_test_energy = pd.read_hdf(join(input_dir, 'energy_test.h5'), 'data')
    # underscor X means sub-sample containing the RF features only
    df_test_energy_X = pd.read_hdf(join(input_dir, 'energy_test_X.h5'), 'data')
    if args.electron:

        print("Reading electrons !!!")
        df_electron = pd.read_hdf(join(input_dir, 'electron.h5'), 'data')
    features = df_test_energy_X.columns
    del df_test_energy_X

    df_test_proton = df_test_classifier.copy()
    mask = df_test_proton['particle'] == 1
    index = np.arange(len(mask))[mask]
    df_test_proton = df_test_proton.iloc[index]
    del df_test_classifier

    # This are gamma on axis
    df_test_energy = add_reconstructed_params(df_test_energy,
                                              df_test_energy[features],
                                              rf_classifier=rf_classifier,
                                              rf_disp=rf_disp,
                                              rf_energy=rf_energy,
                                              focal=focal,
                                              source_x=source_x,
                                              source_y=source_y,
                                              n_telescopes=n_telescopes)
    df_test_proton = add_reconstructed_params(df_test_proton,
                                              df_test_proton[features],
                                              rf_classifier=rf_classifier,
                                              rf_disp=rf_disp,
                                              rf_energy=rf_energy,
                                              focal=focal,
                                              source_x=0, source_y=0,
                                              n_telescopes=n_telescopes)

    if args.electron:

        df_electron = add_reconstructed_params(df_electron,
                                               df_electron[features],
                                               rf_classifier=rf_classifier,
                                               rf_disp=rf_disp,
                                               rf_energy=rf_energy,
                                               focal=focal,
                                               source_x=0, source_y=0,
                                               n_telescopes=n_telescopes)

    gamma_cuts = np.linspace(0, 1, num=21)

    if args.theta is None:
        theta_2_68 = np.quantile(df_test_energy['theta_2'], 0.68)
        theta_min = 0  # 0.000001523
        theta_max = theta_2_68  # 0.000015231
        theta_2_cuts = np.linspace(theta_min, theta_max, num=20)

    else:
        theta_2_cuts = np.array([np.radians(float(args.theta))**2])
        theta_min = 0
        theta_max = theta_2_cuts[0]

    sensitivity = np.ones((len(gamma_cuts), len(theta_2_cuts),
                           len(bins_energy_CTA_width))) * np.inf
    sensitivity_err = np.zeros(
        (len(gamma_cuts), len(theta_2_cuts), len(bins_energy_CTA_width)))
    sensitivity_err_min = np.zeros(
        (len(gamma_cuts), len(theta_2_cuts), len(bins_energy_CTA_width)))
    sensitivity_err_max = np.zeros(
        (len(gamma_cuts), len(theta_2_cuts), len(bins_energy_CTA_width)))

    index_gamma_cut = [None] * len(bins_energy_CTA_mid)
    index_theta_2_cut = [None] * len(bins_energy_CTA_mid)
    T_obs = 50 * 60 * 60
    alpha = 1 / 5
    theta_2_proton_max = np.radians(1.5) ** 2  # 1 deg**2
    omega_proton = 2 * np.pi * (1 - np.cos(np.sqrt(theta_2_proton_max)))
    omega_theta_2_cut = 2 * np.pi * (1 - np.cos(np.sqrt(theta_2_cuts)))

    for i, gamma_cut in tqdm(enumerate(gamma_cuts)):
        for j, theta_2_cut in tqdm(enumerate(theta_2_cuts)):
            n_p, n_g = len(df_test_proton), len(df_test_energy)
            mask_proton = (df_test_proton['theta_2'] <= theta_2_proton_max) & (
                        df_test_proton['gammaness'] >= gamma_cut)
            mask_gamma = (df_test_energy['theta_2'] <= theta_2_cut) & (
                        df_test_energy['gammaness'] >= gamma_cut)

            if args.electron:
                mask_electron = (df_electron['theta_2'] <= theta_2_proton_max) & (df_electron['gammaness'] >= gamma_cut)



            df_p = df_test_proton.loc[mask_proton]
            df_g = df_test_energy.loc[mask_gamma]

            w_p = df_p['weights'] / test_size * T_obs / omega_proton * \
                  omega_theta_2_cut[j] / alpha
            w_g = df_g['weights'] * T_obs
            n_p, _ = np.histogram(np.log10(df_p['reco_energy']),
                                  bins=bins_energy_CTA, weights=w_p)
            n_g, _ = np.histogram(np.log10(df_g['reco_energy']),
                                  bins=bins_energy_CTA, weights=w_g)

            n_p_error = np.sqrt(
                np.histogram(np.log10(df_p['reco_energy']), bins=bins_energy_CTA,
                             weights=w_p ** 2)[0])
            n_g_error = np.sqrt(
                np.histogram(np.log10(df_g['reco_energy']), bins=bins_energy_CTA,
                             weights=w_g ** 2)[0])

            n_p_sim, _ = np.histogram(np.log10(df_p['reco_energy']),
                                      bins=bins_energy_CTA)
            n_g_sim, _ = np.histogram(np.log10(df_g['reco_energy']),
                                      bins=bins_energy_CTA)
            mask = (n_p_sim >= 5) & (n_g_sim >= 5)

            if args.electron:

                df_e = df_electron.loc[mask_electron]
                w_e = df_e['weights'] * T_obs / omega_proton / alpha * omega_theta_2_cut[j]
                n_e, _ = np.histogram(np.log10(df_e['reco_energy']),
                                  bins=bins_energy_CTA, weights=w_e)
                n_e_sim, _ = np.histogram(np.log10(df_e['reco_energy']),
                                  bins=bins_energy_CTA)
                n_e_error = np.sqrt(np.histogram(np.log10(df_e['reco_energy']),
                                  bins=bins_energy_CTA, weights=w_e**2)[0])
                n_p = n_p + n_e
                n_p_error = np.sqrt(n_p_error**2 + n_e_error**2)

                mask = mask & (n_e_sim >= 0) # Normally would use 5 but satistic for electrons is limited and does not affect much sensitivity

            sigma, sigma_err, sigma_min, sigma_max = compute_sigma(n_g, n_p, T_obs,
                                                                   T_obs / alpha,
                                                                   n_g_err=n_g_error,
                                                                   n_p_err=n_p_error)
            sigma[~mask] = np.inf
            sigma_err[~mask] = np.inf
            # print((n_g / T_on)/ (n_p / T_off))
            sensitivity[i, j] = sigma * crab_spectrum(bins_energy_CTA_mid)
            sensitivity_err[i, j] = sigma_err * crab_spectrum(bins_energy_CTA_mid)
            sensitivity_err_min[i, j] = sigma_min * crab_spectrum(
                bins_energy_CTA_mid)
            sensitivity_err_max[i, j] = sigma_max * crab_spectrum(
                bins_energy_CTA_mid)

    differential_sensitivity = np.nanmin(sensitivity, axis=(0, 1))
    differential_sensitivity_err = np.zeros(differential_sensitivity.shape)
    differential_sensitivity_err_min = np.zeros(differential_sensitivity.shape)
    differential_sensitivity_err_max = np.zeros(differential_sensitivity.shape)

    for k in range(sensitivity.shape[-1]):
        A = sensitivity[:, :, k]
        i, j = np.unravel_index(A.argmin(), A.shape)
        index_gamma_cut[k] = i
        index_theta_2_cut[k] = j
        differential_sensitivity_err[k] = sensitivity_err[i, j, k]
        differential_sensitivity_err_min[k] = sensitivity_err_min[i, j, k]
        differential_sensitivity_err_max[k] = sensitivity_err_max[i, j, k]

    theta_cuts_bins = theta_2_cuts.copy()
    gamma_cuts_bins = gamma_cuts.copy()
    theta_2_cuts = theta_2_cuts[index_theta_2_cut]
    gamma_cuts = gamma_cuts[index_gamma_cut]
    mask_p = np.zeros(len(df_test_proton), dtype=bool)
    mask_g = np.zeros(len(df_test_energy), dtype=bool)

    weights_p = np.array(
        df_test_proton['weights']) * T_obs / alpha / omega_proton / test_size
    weights_g = np.array(df_test_energy['weights']) * T_obs

    if args.electron:
        mask_e = np.zeros(len(df_electron), dtype=bool)
        weights_e = np.array(df_electron['weights']) * T_obs / omega_proton / alpha

    for i in range(len(bins_energy_CTA_width)):
        mask = (np.log10(df_test_proton['reco_energy']) <= bins_energy_CTA[
            i + 1]) & (np.log10(df_test_proton['reco_energy']) >
                       bins_energy_CTA[i]) & (
                           df_test_proton['theta_2'] <= theta_2_proton_max) & (
                           df_test_proton['gammaness'] >= gamma_cuts[i])
trainenr
        mask_p += mask
        mask_g += (np.log10(df_test_energy['reco_energy']) <= bins_energy_CTA[
            i + 1]) & (np.log10(df_test_energy['reco_energy']) >
                       bins_energy_CTA[i]) & (
                              df_test_energy['theta_2'] <= theta_2_cuts[i]) & (
                              df_test_energy['gammaness'] >= gamma_cuts[i])
        weights_p[mask] = weights_p[mask] * (
                    2 * np.pi * (1 - np.cos(np.sqrt(theta_2_cuts[i]))))

        if args.electron:

            mask = (np.log10(df_electron['reco_energy']) <= bins_energy_CTA[
            i + 1]) & (np.log10(df_electron['reco_energy']) >
                       bins_energy_CTA[i]) & (
                              df_electron['theta_2'] <= theta_2_proton_max) & (
                              df_electron['gammaness'] >= gamma_cuts[i])
            mask_e += mask
            weights_e[mask] = weights_e[mask] *  (
                    2 * np.pi * (1 - np.cos(np.sqrt(theta_2_cuts[i]))))

    mask_p = np.arange(len(mask_p))[mask_p]
    mask_g = np.arange(len(mask_g))[mask_g]
    df_p = df_test_proton.iloc[mask_p]
    df_g = df_test_energy.iloc[mask_g]
    w_p = weights_p[mask_p]
    w_g = weights_g[mask_g]

    df_p['weights'] = w_p
    df_g['weights'] = w_g
    df_p.to_hdf(join(output_dir, 'dl2_proton.h5'), key='data')
    df_g.to_hdf(join(output_dir, 'dl2_gamma.h5'), key='data')


    def save_dl2(df, output_file, gamma_cuts, bins_energy=bins_energy_CTA):

        mask = np.zeros(len(df), dtype=bool)
        index = np.arange(len(df))
        for i in range(len(bins_energy)-1):
            mask += (np.log10(df['reco_energy']) <= bins_energy[
                i + 1]) & (np.log10(df['reco_energy']) >
                           bins_energy[i]) & (
                           df['gammaness'] >= gamma_cuts[i])

        df = df.iloc[index[mask]]
        df.to_hdf(output_file, key='data')

        return df

    save_dl2(df_test_energy, join(output_dir, 'dl2_gamma_gammacut_only.h5'), gamma_cuts=gamma_cuts, bins_energy=bins_energy_CTA)
    save_dl2(df_test_proton, join(output_dir, 'dl2_proton_gammacut_only.h5'), gamma_cuts=gamma_cuts, bins_energy=bins_energy_CTA)

    if args.electron:

        mask_e = np.arange(len(mask_e))[mask_e]
        df_e = df_electron.iloc[mask_e]
        w_e = weights_e[mask_e]
        df_e.to_hdf(join(output_dir, 'dl2_electron.h5'), key='data')
        save_dl2(df_electron,
                 join(output_dir, 'dl2_electron_gammacut_only.h5'),
                 gamma_cuts=gamma_cuts, bins_energy=bins_energy_CTA)

    for energy_kind in ['reco_energy', 'true_energy']:
        n_p, _ = np.histogram(np.log10(df_p[energy_kind]), bins=bins_energy_CTA,
                              weights=w_p * alpha)
        n_g, _ = np.histogram(np.log10(df_g[energy_kind]), bins=bins_energy_CTA,
                              weights=w_g)
        n_p_err = np.sqrt(
            np.histogram(np.log10(df_p[energy_kind]), bins=bins_energy_CTA,
                         weights=(w_p * alpha) ** 2)[0])
        n_g_err = np.sqrt(
            np.histogram(np.log10(df_g[energy_kind]), bins=bins_energy_CTA,
                         weights=w_g ** 2)[0])

        if args.electron:

            n_e, _ = np.histogram(np.log10(df_e[energy_kind]), bins=bins_energy_CTA,
                                  weights=w_e * alpha)
            n_e_err = np.sqrt(np.histogram(np.log10(df_e[energy_kind]), bins=bins_energy_CTA,
                                  weights=(w_e * alpha)**2)[0])

        plt.figure(figsize=figsize)
        plt.errorbar(bins_energy_CTA_mid, n_p, yerr=n_p_err, marker='o',
                     color='r', label=r'Proton, $N_p = \alpha N_{off}$',
                     linestyle='None')
        plt.errorbar(bins_energy_CTA_mid, n_g, yerr=n_g_err, marker='o',
                     color='g',
                     label=r'Gamma, $N_{\gamma} = N_{on} - \alpha N_{off}$',
                     linestyle='None')
        plt.step(bins_energy_CTA_mid, n_p, color='r', where='mid')
        plt.step(bins_energy_CTA_mid, n_g, color='g', where='mid')

        if args.electron:

            plt.errorbar(bins_energy_CTA_mid, n_e, yerr=n_e_err, marker='o',
                         color='k', label='Electron, $N_e$', linestyle='None',
                         )
            plt.step(bins_energy_CTA_mid, n_e, color='k', where='mid')

        plt.yscale('log')
        plt.xscale('log')
        plt.xlabel(LABELS[energy_kind])
        plt.ylabel('Re-weighted counts (50 h) []')
        plt.legend()
        plt.savefig(join(output_dir,
                         'events_after_cuts_reweigthed_{}.pdf'.format(
                             energy_kind)))

        plt.figure(figsize=figsize)
        plt.hist(df_p[energy_kind], bins=10 ** bins_energy_CTA, label='Proton',
                 histtype='step', color='r')
        plt.hist(df_g[energy_kind], bins=10 ** bins_energy_CTA, label='Gamma',
                 histtype='step', color='g')

        if args.electron:
            plt.hist(df_e[energy_kind], bins=10**bins_energy_CTA,
                     label='Electron', histtype='step', color='k')

        plt.yscale('log')
        plt.xscale('log')
        plt.xlabel(LABELS[energy_kind])
        plt.ylabel('Simulated counts []')
        plt.legend()
        plt.savefig(join(output_dir,
                         'events_after_cuts_simulated_{}.pdf'.format(
                             energy_kind)))

    fig = plt.figure(figsize=figsize)
    axes = fig.add_subplot(111)
    axes.errorbar(bins_energy_CTA_mid, theta_2_cuts,
                  xerr=bins_energy_CTA_width * 0.5, linestyle='None',
                  marker='o', color='k')
    axes.set_xscale('log')
    axes.set_ylim(theta_min, theta_2_cuts.max())
    axes.set_ylabel(r'$\theta^2$ cut [rad$^2$]')
    axes.set_xlabel(LABELS['reco_energy'])
    fig.savefig(join(output_dir, 'sensitivity_theta2_cuts.pdf').format(i))

    fig = plt.figure(figsize=figsize)
    axes = fig.add_subplot(111)
    axes.errorbar(bins_energy_CTA_mid, gamma_cuts,
                  xerr=bins_energy_CTA_width * 0.5, linestyle='None',
                  marker='o', color='k')
    axes.set_xscale('log')
    axes.set_ylim(0, 1)
    axes.set_ylabel('Gammaness cut []')
    axes.set_xlabel(LABELS['reco_energy'])
    fig.savefig(join(output_dir, 'sensitivity_gammaness_cuts.pdf').format(i))

    e_smooth = np.logspace(np.log10(bins_energy_CTA_mid.min()),
                           np.log10(bins_energy_CTA_mid.max()), 1000)

    factors = [1.60218 * 1E-4, 1]
    y_labels = ['$\mathsf{E^2 F \; [erg \, cm^{-2} s^{-1}]}$',
                '$\mathsf{E^2 F \; [TeV \, m^{-2} s^{-1}]}$']

    data = np.array([bins_energy_CTA[:-1], bins_energy_CTA[1:], differential_sensitivity, differential_sensitivity_err, theta_2_cuts, gamma_cuts])
    np.savetxt(join(output_dir, 'sensitivity.txt'), data.T, comments='1. log10(e-bin) min 2. log10(e-bin) max 3. sensitivity 4. sensitivity err 5. theta² cut 6. gamma-ness cut')

    for i, factor in enumerate(factors):

        fig = plt.figure(figsize=(12, 8))
        axes = fig.add_subplot(111)
        y_data = factor * (bins_energy_CTA_mid) ** 2 * differential_sensitivity
        mask = np.isfinite(y_data)
        axes.errorbar(bins_energy_CTA_mid, y_data,
                      color='red', xerr=0.5 * bins_energy_CTA_width,
                      yerr=factor * (
                          bins_energy_CTA_mid) ** 2 * differential_sensitivity_err,
                      linestyle='None', marker='o',
                      label='(this work)')
        y = factor * e_smooth ** 2 * crab_spectrum(e_smooth)
        axes.plot(e_smooth, y, '-',
                  alpha=0.5, color='grey', label='100% Crab')  # 100 %
        axes.plot(e_smooth, y * 0.1, '--',
                  alpha=0.5, color='grey', label='10% Crab')  # 10 %
        # axes.plot(e_smooth, y * 0.01, '-.',
        #          alpha=0.5, color='grey', label='1% Crab')  # 10 %
        axes.set_xscale('log')
        axes.set_yscale('log')
        axes.set_xlim(10 ** np.floor(bins_energy_CTA[:-1][mask].min()),
                      10 ** np.ceil(bins_energy_CTA[:-1][mask].max()))
        axes.set_ylim(10 ** np.floor(np.min(np.log10(y_data))),
                      10 ** np.ceil(np.max(np.log10(y)) + 1))
        axes.set_ylabel(y_labels[i])
        axes.set_xlabel(LABELS['reco_energy'])
        axes.grid(True, which='both')
        axes.legend(loc='best')
        fig.savefig(join(output_dir, 'sensitivity_{}.pdf').format(i))
