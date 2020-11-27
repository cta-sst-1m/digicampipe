
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from digicampipe.visualization.machine_learning import plot_feature_importance, \
    plot_confusion_matrix, plot_roc_curve, plot_roc_curve_energy, \
    plot_2d_source, plot_2d_disp, plot_delta_disp, plot_r68, plot_delta_energy, \
    plot_resolution_energy, plot_classifier_distribution, plot_features, plot_init_vs_fit, get_effective_area_weights, plot_grid_cv, plot_grid_cv_2, plot_3d_histo, plot_differential_trigger_rate, plot_effective_area, LABELS, compute_differential_trigger_rate, crab_spectrum, get_rate_gamma, get_rate_proton
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import roc_auc_score, \
    r2_score
from sklearn.model_selection import \
    GridSearchCV  # Create the parameter grid based on the results of random search
import joblib
import pickle as pk

from digicampipe.io.dl1 import read_hdf5, combine_datasets, read_event_histogram
from digicampipe.scripts.effective_area import make_3d_histogram, get_fitted
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd
from digicampipe.pointing.disp import cal_cam_source_pos
from digicampipe.image.hillas import compute_alpha
from tqdm import tqdm
import argparse


def sigma_lima(n_on, n_off, alpha):
    sigma_lima = np.sqrt(2 * (n_on * np.log((1 + alpha) / alpha * n_on / (n_on + n_off)) + n_off * np.log((1 + alpha) * (n_off / (n_on + n_off)))))
    return sigma_lima

def quality_cuts(df, intensity_cut=15, leakage_cut=(0.6, 1.5), wol_cut=(0, np.inf)):

    n_events = len(df)
    mask = (df['intensity'] >= intensity_cut) & (df['leakage'] <= leakage_cut[1]) & (df['leakage'] >= leakage_cut[0]) # & (df['r'] <= 380)
    mask = mask & (df['wol'] > wol_cut[0]) & (df['wol'] <= wol_cut[1])
    mask = np.arange(len(df))[mask]
    df_out = df.iloc[mask]
    with pd.option_context('mode.use_inf_as_null', True): # Drops infinite values
        df_out = df_out.dropna()

    n_dropped = n_events - len(df_out)
    return df_out

def get_data(file, max_events=None, intensity_cut=0, leakage_cut=(0., np.inf), wol_cut=(0.1, 1), kind='fit'):

    df_triggered = pd.read_hdf(file, key='data',)

    if max_events is not None:
        df_triggered = df_triggered.sample(frac=1)
        max_events = min(len(df_triggered), max_events)
        df_triggered = df_triggered.iloc[:max_events]

    """
    df_triggered['log_lh'] = np.zeros(len(df_triggered))

    if 'density' not in df_triggered.columns:

        df_triggered['density'] = df_triggered['intensity'] / df_triggered['area']
        df_triggered['density_w'] = df_triggered['intensity'] / df_triggered['width']
        df_triggered['density_l'] = df_triggered['intensity'] / df_triggered['length']
        df_triggered['kurtosis_w'] = df_triggered['kurtosis']
        df_triggered['skewness_l'] = df_triggered['skewness']
        df_triggered['true_energy'] = df_triggered['energy']
    """
    df_fitted = get_fitted(df_triggered)

    df_quality = quality_cuts(df_fitted, intensity_cut=intensity_cut, leakage_cut=leakage_cut, wol_cut=wol_cut)

    print("Sample loaded with simulated, {} triggered, {} fitted remaining {} after quality cut".format(
        len(df_triggered),
            len(df_fitted), len(df_quality)))

    return df_quality


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Process some value.')
    parser.add_argument('--max_events', default=None, type=int,
                        help='sum the integers (default: find the max)')
    parser.add_argument('--size_cut', default=50,
                        type=float, help="A cut on size")
    parser.add_argument('--gamma_cut', default=None,
                        type=float, help="Threshold to rejected protons")
    parser.add_argument('--theta2_cut', default=None,
                        type=float, help="Threshold to cut of the angular distance square")
    parser.add_argument('--cv', default=2,
                        type=int, help='Number of cross-validation')
    parser.add_argument('--proton',  default='/sst1m/analyzed/mpeshowerfit/tmp/zenith_20_proton_v15.hdf5',
                        type=str, help='Processing version')
    parser.add_argument('--gamma', default='/sst1m/analyzed/mpeshowerfit/tmp/zenith_20_gamma_v15.hdf5',
                        type=str, help='Processing version')
    parser.add_argument('--gamma_diffuse',default='/sst1m/analyzed/mpeshowerfit/zenith_20_diffuse_gamma_v15.hdf5',
                        type=str, help='Processing version')
    parser.add_argument('--electron', help='path to electron file',
                        type=str, default=None)
    parser.add_argument('--output_directory', default='.',
                        type=str, help='Processing version')
    parser.add_argument('--kind', default='fit',
                        type=str, help='use the fit values or the init values')
    parser.add_argument('--focal', default=5600,
                        type=float, help='Focal length of the telescope')
    parser.add_argument('--source_x', default=0,
                        type=float, help='Source position in x [mm]')
    parser.add_argument('--source_y', default=0,
                        type=float, help='Source position in y [mm]')
    parser.add_argument('--leakage_min', default=0,
                        type=float, help='Leakage cut min')
    parser.add_argument('--leakage_max', default=np.inf,
                        type=float, help='Leakage cut max')
    parser.add_argument('--wol_min', default=0.1,
                        type=float, help='Minimum width over length')
    parser.add_argument('--wol_max', default=np.inf,
                        type=float, help='Maximum width over length')


    args = parser.parse_args()

    print(args)

    test_size = 0.65
    max_events = args.max_events # None # 1000 # 10000
    FOCAL = args.focal

    cv = args.cv
    max_depth_energy = [18, 22, 26, None] #16, 18, 20]
    # n_estimators_energy = [52, 56, 60, 64, 68, 72] # [50, 75, 100, 125, 150, 175, 200, 225, 250, 275, 300]
    n_estimators_energy = [50, 60, 70,] # [50, 75, 100, 125, 150, 175, 200, 225, 250, 275, 300]
    parameters_energy = {'n_estimators': n_estimators_energy, 'max_depth': max_depth_energy}
    # parameters_energy = {'n_estimators': [10], 'max_depth': [10]}
    max_depth_disp = [18, 22, 26, None] #16, 18, 20]
    n_estimators_disp = [50, 60, 70,]# [50, 75, 100, 125, 150, 175, 200, 225, 250, 275, 300]
    parameters_disp = {'n_estimators': n_estimators_disp, 'max_depth': max_depth_disp}
    # parameters_disp = {'n_estimators': [10], 'max_depth': [10]}
    max_depth_classifier = [12, 14, 16, None] #, 5, 7, 9, 11, 13, 15 ]# 11, 13, 15, 17, 19,]
    n_estimators_classifier = [30, 40, 50, ] # [10, 15, 20, 25, 30, 40] # 35, 40,] # [10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60] # [20, 25, 30, ]
    parameters_classifier = {'n_estimators': n_estimators_classifier, 'max_depth': max_depth_classifier}
    # parameters_classifier = {'n_estimators': [10], 'max_depth': [10]}
    size_cut = args.size_cut
    gamma_cut = args.gamma_cut
    wol_cut = (args.wol_min, args.wol_max)
    leakage_cut = (args.leakage_min, args.leakage_max)
    target_disp = ['disp_x', 'disp_y']
    # target_disp = ['disp_theta', 'disp_r']
    kind = args.kind

    if gamma_cut is None:
        gamma_cuts = np.linspace(0, 1, num=100)
    else:
        gamma_cuts = np.array(gamma_cut)

    theta_2_cut = args.theta2_cut
    if theta_2_cut is None:

        theta_2_cuts = np.radians(np.linspace(0, 4, num=400))**2
    else:
        theta_2_cuts = np.array(theta_2_cut)
    verbose = 3

    output_dir = args.output_directory

    gamma_file = args.gamma
    proton_file = args.proton
    gamma_diffuse_file = args.gamma_diffuse
    electron_diffuse_file = args.electron

    print("Reading gamma sample")
    df_gamma_quality = get_data(file=gamma_file, max_events=max_events, leakage_cut=leakage_cut, wol_cut=wol_cut, intensity_cut=size_cut, kind=kind)
    df_gamma_quality['particle'] = np.zeros(len(df_gamma_quality))

    print('Total rate after quality cut gamma {:.2f} [mHz]'.format(np.sum(df_gamma_quality['weights'])*1E3))
    print("Reading proton sample")
    df_proton_quality = get_data(file=proton_file, max_events=max_events, leakage_cut=leakage_cut, wol_cut=wol_cut, intensity_cut=size_cut, kind=kind)
    df_proton_quality['particle'] = np.ones(len(df_proton_quality))

    print('Total rate after quality cut proton {:.2f} [Hz]'.format(np.sum(df_proton_quality['weights'])))
    print("Reading gamma diffuse sample")
    df_gamma_diffuse_quality = get_data(file=gamma_diffuse_file, max_events=max_events, leakage_cut=leakage_cut, wol_cut=wol_cut, intensity_cut=size_cut, kind=kind)
    df_gamma_diffuse_quality['particle'] = np.zeros(len(df_gamma_diffuse_quality))

    print("Reading electron diffuse sample")
    df_electron_quality = get_data(file=electron_diffuse_file, max_events=max_events, leakage_cut=leakage_cut, wol_cut=wol_cut, intensity_cut=size_cut, kind=kind)
    print("Total rate after quality cut electron {:.6f} [Hz]".format(np.sum(df_electron_quality['weights'])))
    df_electron_quality.to_hdf(os.path.join(output_dir, 'electron.h5'), key='data')

    fit_params = ['alpha', 'intensity', 'length', 'width', 'phi', 'psi',
                  'r', 'x', 'y', 'slope', 'intercept']

    figure_path = os.path.join(output_dir, 'figure_parameters.pdf')

    # target_features = ['particle', 'true_energy', 'disp_x', 'disp_y']
    train_features = ['intensity', 'kurtosis_w',
                      'length', 'psi', 'skewness_l', 'width',
                      'slope', 'area', 't_68',
                      'r', 'x', 'y', 'phi',
                      'leakage',
                      #'density',
                      # 'density_l',
                      # 'density_w',
                      ]


    df_proton_train, df_proton_test = train_test_split(df_proton_quality, test_size=test_size)
    # df_gamma_diffuse_train, df_gamma_diffuse_test = train_test_split(df_gamma_diffuse_quality, test_size=test_size)

    df_train_classifier = combine_datasets(df_proton_train, df_gamma_diffuse_quality)
    df_test_classifier = combine_datasets(df_proton_test, df_gamma_quality)
    print('HELLO ')
    print(len(df_proton_train), len(df_proton_test), len(df_proton_quality), len(df_gamma_quality), len(df_gamma_diffuse_quality))
    df_test_classifier.to_hdf(os.path.join(output_dir, 'classifier_test.h5'), 'data')
    df_train_classifier.to_hdf(os.path.join(output_dir, 'classifier_train.h5'), 'data')
    df_train_classifier_X, df_train_classifier_y = df_train_classifier[train_features], df_train_classifier['particle']
    df_test_classifier_X, df_test_classifier_y = df_test_classifier[train_features], df_test_classifier['particle']
    df_test_classifier_X.to_hdf(os.path.join(output_dir, 'classifier_test_X.h5'), 'data')
    df_train_classifier_X.to_hdf(os.path.join(output_dir, 'classifier_train_X.h5'), 'data')
    df_test_classifier_y.to_hdf(os.path.join(output_dir, 'classifier_test_y.h5'), 'data')
    df_train_classifier_y.to_hdf(os.path.join(output_dir, 'classifier_train_y.h5'), 'data')

    df_train_energy = df_gamma_diffuse_quality
    df_test_energy = df_gamma_quality
    df_train_energy.to_hdf(os.path.join(output_dir, 'energy_train.h5'), 'data')
    df_test_energy.to_hdf(os.path.join(output_dir, 'energy_test.h5'), 'data')
    df_train_energy_X, df_train_energy_y = df_train_energy[train_features], df_train_energy['true_energy']
    df_test_energy_X, df_test_energy_y = df_test_energy[train_features], df_test_energy['true_energy']
    df_train_energy_y.to_hdf(os.path.join(output_dir, 'energy_train_y.h5'), 'data')
    df_train_energy_X.to_hdf(os.path.join(output_dir, 'energy_train_X.h5'), 'data')
    df_test_energy_X.to_hdf(os.path.join(output_dir, 'energy_test_X.h5'), 'data')
    df_test_energy_y.to_hdf(os.path.join(output_dir, 'energy_test_y.h5'), 'data')

    df_train_disp = df_gamma_diffuse_quality
    df_test_disp = df_gamma_quality
    df_train_disp.to_hdf(os.path.join(output_dir, 'disp_train.h5'), 'data')
    df_test_disp.to_hdf(os.path.join(output_dir, 'disp_test.h5'), 'data')
    df_train_disp_X, df_train_disp_y = df_train_disp[train_features], df_train_disp[target_disp]
    df_test_disp_X, df_test_disp_y = df_test_disp[train_features], df_test_disp[target_disp]
    df_train_disp_X.to_hdf(os.path.join(output_dir, 'disp_train_X.h5'), 'data')
    df_test_disp_X.to_hdf(os.path.join(output_dir, 'disp_test_X.h5'), 'data')
    df_test_disp_y.to_hdf(os.path.join(output_dir, 'disp_test_y.h5'), 'data')
    df_train_disp_y.to_hdf(os.path.join(output_dir, 'disp_train_y.h5'), 'data')



    features = df_test_classifier_X.columns

    print("Training energy reconstruction")
    print("Grid search for {} with {:d} cross-validations".format(parameters_energy, cv))
    grid_result_energy = GridSearchCV(RandomForestRegressor(n_jobs=-1, verbose=verbose), parameters_energy, cv=cv, return_train_score=True)
    grid_result_energy.fit(df_train_energy_X, df_train_energy_y)
    results_energy = grid_result_energy.cv_results_
    print(results_energy, parameters_energy)
    rf_energy = grid_result_energy.best_estimator_
    best_params_energy = grid_result_energy.best_params_
    joblib.dump(rf_energy, os.path.join(output_dir, 'energy_regressor.sav'))
    pk.dump(results_energy, open(os.path.join(output_dir, 'gridcv_energy_regressor.pk'), 'wb'))

    print("Random Forest energy-regressor fitted !")
    print("Best parameters : {}".format(best_params_energy))
    prediction_energy_test = rf_energy.predict(df_test_energy_X)
    prediction_energy_train = rf_energy.predict(df_train_energy_X)
    r2_energy_test = rf_energy.score(df_test_energy_X, df_test_energy_y)
    r2_energy_train = rf_energy.score(df_train_energy_X, df_train_energy_y)
    print("R2 score : {:.6f} (Test)\t {:.6f} (Train)".format(r2_energy_test, r2_energy_train))
    r2_energy_test = r2_score(df_test_energy['true_energy'], prediction_energy_test)
    r2_energy_train = r2_score(df_train_energy['true_energy'], prediction_energy_train)
    print('R2 score Energy : {:.6f} (Test)\t {:.6f} (Train)'.format(r2_energy_test, r2_energy_train))
    importance_energy = rf_energy.feature_importances_

    print("Training DISP reconstruction")
    print("Grid search for {} with {:d} cross-validations".format(parameters_disp, cv))
    grid_result_disp = GridSearchCV(RandomForestRegressor(n_jobs=-1, verbose=verbose), parameters_disp, cv=cv, return_train_score=True)
    grid_result_disp.fit(df_train_disp_X, df_train_disp_y)
    results_disp = grid_result_disp.cv_results_
    print(results_disp, parameters_disp)
    rf_disp = grid_result_disp.best_estimator_
    importance_disp = rf_disp.feature_importances_
    best_params_disp = grid_result_disp.best_params_
    joblib.dump(rf_disp, os.path.join(output_dir, 'disp_regressor.sav'))
    pk.dump(results_disp, open(os.path.join(output_dir, 'gridcv_disp_regressor.pk'), 'wb'))
    print("Random Forest DISP-regressor fitted !")
    print("Best parameters : {}".format(best_params_disp))
    prediction_disp_test = rf_disp.predict(df_test_disp_X)
    prediction_disp_train = rf_disp.predict(df_train_disp_X)
    r2_direction_test = r2_score(df_test_disp_y, prediction_disp_test)
    r2_direction_test_x = r2_score(df_test_disp[target_disp[0]], prediction_disp_test[:, 0])
    r2_direction_test_y = r2_score(df_test_disp[target_disp[1]], prediction_disp_test[:, 1])
    r2_direction_train = r2_score(df_train_disp_y, prediction_disp_train)
    r2_direction_train_x = r2_score(df_train_disp[target_disp[0]], prediction_disp_train[:, 0])
    r2_direction_train_y = r2_score(df_train_disp[target_disp[1]], prediction_disp_train[:, 1])

    print('R2 score Direction : {:.6f} (Test)\t {:.6f} (Train)'.format(
        r2_direction_test, r2_direction_train))
    print('R2 score Direction {}: {:.6f} (Test)\t {:.6f} (Train)'.format(
        target_disp[0], r2_direction_test_x, r2_direction_train_x))
    print('R2 score Direction {}: {:.6f} (Test)\t {:.6f} (Train)'.format(
        target_disp[1], r2_direction_test_y, r2_direction_train_y))

    if target_disp == ['disp_theta', 'disp_r']:
        prediction_disp_test[:, 0], prediction_disp_test[:, 1] = prediction_disp_test[:, 1] * np.cos(prediction_disp_test[:, 0]), prediction_disp_test[:, 1] * np.sin(prediction_disp_test[:, 0])
        prediction_disp_train[:, 0], prediction_disp_train[:, 1]  = prediction_disp_train[:, 1] * np.cos(prediction_disp_train[:, 0]), prediction_disp_train[:, 1] * np.sin(prediction_disp_train[:, 0])


    print("Training classification")
    print("Grid search for {} with {:d} cross-validations".format(parameters_classifier, cv))
    grid_result_classifier = GridSearchCV(RandomForestClassifier(n_jobs=-1, verbose=verbose), parameters_classifier, cv=cv, scoring='roc_auc', return_train_score=True)
    grid_result_classifier.fit(df_train_classifier_X, df_train_classifier_y)
    results_classifier = grid_result_classifier.cv_results_
    print(results_classifier, parameters_classifier)
    rf_classifier = grid_result_classifier.best_estimator_
    best_params_classifier = grid_result_classifier.best_params_
    joblib.dump(rf_classifier, os.path.join(output_dir, 'classifier.sav'))
    pk.dump(results_classifier, open(os.path.join(output_dir, 'gridcv_classifier.pk'), 'wb'))
    print("Random Forest Classifier fitted !")
    print("Best parameters : {}".format(best_params_classifier))
    prediction_classifier_test = rf_classifier.predict_proba(df_test_classifier_X)
    prediction_classifier_train = rf_classifier.predict_proba(df_train_classifier_X)
    accuracy_particle_test = rf_classifier.score(df_test_classifier_X, df_test_classifier_y)
    accuracy_particle_train = rf_classifier.score(df_train_classifier_X, df_train_classifier_y)
    print("Mean accuracy : {:.6f} (Test)\t {:.6f} (Train)".format(accuracy_particle_test, accuracy_particle_train))
    roc_auc_score_test = roc_auc_score(df_test_classifier_y, prediction_classifier_test[:, 1])
    roc_auc_score_train = roc_auc_score(df_train_classifier_y, prediction_classifier_train[:, 1])
    print('ROC AUC : {:.6f} (Test)\t {:.6f} (Train)'.format(roc_auc_score_test, roc_auc_score_train))
    importance_classifier = rf_classifier.feature_importances_
    figure_path = os.path.join(output_dir, 'figure_regressor.pdf')


    def add_reco_parameters(df, source_x=0, source_y=0):
        df['gammaness'] = rf_classifier.predict_proba(df[train_features])[:, 0]
        X = rf_disp.predict(df[train_features])

        if target_disp == ['disp_theta', 'disp_r']:
            df['reco_disp_theta'] = X[:, 0]
            df['reco_disp_r'] = X[:, 1]

            df['reco_disp_x'] = X[:, 1] * np.cos(X[:, 0])
            df['reco_disp_y'] = X[:, 1] * np.sin(X[:, 0])

        elif target_disp == ['disp_x', 'disp_y']:


            df['reco_disp_x'] = X[:, 0]
            df['reco_disp_y'] = X[:, 1]
            df['reco_disp_theta'] = np.arctan2(X[:, 1], X[:, 0])
            df['reco_disp_r'] = np.sqrt(X[:, 0]**2 + X[:, 1]**2)

        df['reco_energy'] = rf_energy.predict(df[train_features])
        df['reco_source_x'] = df['reco_disp_x'] + df['x']
        df['reco_source_y'] = df['reco_disp_y'] + df['y']
        df['theta_2'] = ((df['reco_source_x'] - source_x) ** 2 + (
                    df['reco_source_y'] - source_y) ** 2) / FOCAL ** 2
        return df

    df_proton_test = add_reco_parameters(df_proton_test, source_x=args.source_x, source_y=args.source_y)
    df_gamma_quality = add_reco_parameters(df_gamma_quality, source_x=args.source_x, source_y=args.source_y)


    def cut_gammaness(df, gammaness=0.5):

        mask = df['gammaness'] >= gammaness
        index = np.arange(len(df))[mask]
        return df.iloc[index]

    df_proton_classification = cut_gammaness(df_proton_test)
    df_gamma_classification = cut_gammaness(df_gamma_quality)

    T_on = 50 * 60 * 60
    alpha = 0.2
    T_off = T_on / alpha




    bins_energy_CTA = np.linspace(-2, 3, num=5*5 + 1) # 5 bins per energy decade
    bins_energy_CTA_width = np.diff(10**bins_energy_CTA)
    bins_energy_CTA_mid = 10**bins_energy_CTA[:-1] + 0.5 * bins_energy_CTA_width

    e_smooth = np.logspace(np.log10(bins_energy_CTA_mid.min()),
                           np.log10(bins_energy_CTA_mid.max()),
                           len(bins_energy_CTA_mid) * 100)


    def compute_sigma(n_gamma, n_proton, t_on, t_off):

        alpha = t_on / t_off
        sigma_1 = 5/sigma_lima(n_on=n_gamma + alpha*n_proton, n_off=n_proton, alpha=alpha)
        sigma_2 = 10/n_gamma
        sigma_3 = 0.05 * (n_proton*alpha/n_gamma)
        sigma = np.array([sigma_1, sigma_2, sigma_3])
        sigma = np.nanmax(sigma, axis=0)
        return sigma

    sensitivity = np.ones((len(gamma_cuts), len(theta_2_cuts), len(bins_energy_CTA_width))) * np.inf
    index_gamma_cut = [None]*len(bins_energy_CTA_mid)
    index_theta_2_cut = [None]*len(bins_energy_CTA_mid)

    for i, gamma_cut in enumerate(gamma_cuts):
        for j, theta_2_cut in enumerate(theta_2_cuts):

            n_p, n_g = len(df_proton_test), len(df_gamma_quality)
            mask_proton = (df_proton_test['theta_2'] <= theta_2_cut) & (df_proton_test['gammaness'] >= gamma_cut)
            mask_gamma = (df_gamma_quality['theta_2'] <= theta_2_cut) & (df_gamma_quality['gammaness'] >= gamma_cut)
            index_proton = np.arange(n_p)[mask_proton]
            index_gamma = np.arange(n_g)[mask_gamma]
            df_p = df_proton_test.iloc[index_proton]
            df_g = df_gamma_quality.iloc[index_gamma]

            n_p, _ = np.histogram(np.log10(df_p['reco_energy']), bins=bins_energy_CTA, weights=df_p['weights'] / test_size)
            n_g, _ = np.histogram(np.log10(df_g['reco_energy']), bins=bins_energy_CTA, weights=df_g['weights'])
            n_p = n_p * T_off
            n_g = n_g * T_on

            sigma = compute_sigma(n_g, n_p, T_on, T_off)
            # print((n_g / T_on)/ (n_p / T_off))
            sensitivity[i, j] = sigma * crab_spectrum(bins_energy_CTA_mid)

    for k in range(sensitivity.shape[-1]):

        A = sensitivity[:, :, k]
        i, j = np.unravel_index(A.argmin(), A.shape)
        index_gamma_cut[k] = i
        index_theta_2_cut[k] = j

    print(index_gamma_cut, index_theta_2_cut)
    print(gamma_cuts[index_gamma_cut])
    print(theta_2_cuts[index_theta_2_cut])

    differential_sensitivity = np.nanmin(sensitivity, axis=(0, 1))
    bins_theta_2 = np.linspace(0, np.radians(np.sqrt(30))**2, num=40)

    with PdfPages(figure_path) as pdf:


        fig = plt.figure()
        axes = fig.add_subplot(111)

        axes.errorbar(np.log10(bins_energy_CTA_mid), gamma_cuts[index_gamma_cut], xerr=np.diff(bins_energy_CTA))
        axes.set_xlabel(LABELS['log_reco_energy'])
        axes.set_ylabel('Gamma-ness []')
        pdf.savefig(fig)

        fig = plt.figure()
        axes = fig.add_subplot(111)

        axes.errorbar(np.log10(bins_energy_CTA_mid), np.degrees(np.sqrt(theta_2_cuts[index_theta_2_cut])), xerr=np.diff(bins_energy_CTA))
        axes.set_xlabel(LABELS['log_reco_energy'])
        axes.set_ylabel(r'$\sqrt{\theta^2}$ [deg]')
        pdf.savefig(fig)

        fig = plt.figure()
        axes = fig.add_subplot(111)
        rate_p, _ = np.histogram(np.log10(df_proton_quality['true_energy']), bins=bins_energy_CTA,
                  weights=df_proton_quality['weights'],)
        rate_g, _ = np.histogram(np.log10(df_gamma_quality['true_energy']), bins=bins_energy_CTA,
                  weights=df_gamma_quality['weights'],)
        axes.step(np.log10(bins_energy_CTA_mid), rate_p, label='Proton',)
        axes.step(np.log10(bins_energy_CTA_mid), rate_g, label='Gamma',)
        axes.legend(loc='best')
        axes.set_yscale('log')
        axes.set_xlabel(LABELS['log_true_energy'])
        axes.set_ylabel('Rate [Hz]')
        pdf.savefig(fig)

        figsize = (10, 8)
        fig = plt.figure(figsize=figsize)
        axes = fig.add_subplot(111)
        axes.set_title('$\gamma$')
        H = axes.hist2d(df_gamma_quality['disp_theta'], df_gamma_quality['psi'], bins=100, norm=LogNorm())
        axes.set_xlabel(LABELS['disp_theta'])
        axes.set_ylabel(LABELS['psi'])
        fig.colorbar(H[3], label='count')
        pdf.savefig(fig)


        fig = plt.figure()
        axes = fig.add_subplot(111)
        axes.errorbar(bins_energy_CTA_mid, (bins_energy_CTA_mid) ** 2 * differential_sensitivity,
                      color='red', xerr=0.5 * bins_energy_CTA_width,
                      linestyle='None', marker='o', label='SST-1M 50h')
        axes.plot(e_smooth, e_smooth ** 2 * crab_spectrum(e_smooth), '-',
                  alpha=0.5, color='grey', label='100% Crab')  # 100 %
        axes.plot(e_smooth, e_smooth ** 2 * 0.1 * crab_spectrum(e_smooth), '--',
                  alpha=0.5, color='grey', label='10% Crab')  # 10 %
        # axes.plot(e_smooth, e_smooth ** 2 * 0.01 * crab_spectrum(e_smooth),
        #          '-.',
        #          alpha=0.5, color='grey', label='1% Crab')  # 1 %
        axes.set_xscale('log')
        axes.set_yscale('log')
        axes.set_xlim(-2, 3)
        axes.set_ylim(1E-8, 1E-5)
        axes.set_ylabel(
            'Differential sensitivity $\mathsf{E^2 F \; [TeV \, m^{-2} s^{-1}]}$')
        axes.set_xlabel(LABELS['reco_energy'])
        axes.legend(loc='best')
        pdf.savefig(fig)

        axes = plot_grid_cv(results_energy)
        axes.set_title('RF Energy')
        pdf.savefig(axes.get_figure())

        axes = plot_grid_cv(results_disp)
        axes.set_title('RF DISP')
        pdf.savefig(axes.get_figure())

        axes = plot_grid_cv(results_classifier)
        axes.set_title('RF Classifier')
        pdf.savefig(axes.get_figure())

        axes = plot_grid_cv_2(grid_result_energy, params_x=parameters_energy['n_estimators'], params_y=parameters_energy['max_depth'])
        axes.set_title('RF Energy')
        pdf.savefig(axes.get_figure())

        axes = plot_grid_cv_2(grid_result_disp, params_x=parameters_disp['n_estimators'], params_y=parameters_disp['max_depth'])
        axes.set_title('RF DISP')
        pdf.savefig(axes.get_figure())

        axes = plot_grid_cv_2(grid_result_classifier, params_x=parameters_classifier['n_estimators'], params_y=parameters_classifier['max_depth'])
        axes.set_title('RF Classifier')
        pdf.savefig(axes.get_figure())

        axes = plot_feature_importance(features, importance_energy)
        axes.set_title('RF Energy')
        pdf.savefig(axes.get_figure())

        axes = plot_feature_importance(features, importance_disp)
        axes.set_title('RF DISP')
        pdf.savefig(axes.get_figure())


        axes = plot_feature_importance(features, importance_classifier)
        axes.set_title('RF Classifier')
        pdf.savefig(axes.get_figure())

        axes = plot_confusion_matrix(df_train_classifier_y, prediction_classifier_train, [r'$\gamma$', r'$p$'])
        axes.set_title('Train sample')
        pdf.savefig(axes.get_figure())

        axes = plot_confusion_matrix(df_test_classifier_y, prediction_classifier_test, [r'$\gamma$', r'$p$'])
        axes.set_title('Test sample')
        pdf.savefig(axes.get_figure())

        axes = plot_classifier_distribution(prediction_classifier_train, df_train_classifier_y)
        axes.set_title('Train sample')
        pdf.savefig(axes.get_figure())

        axes = plot_classifier_distribution(prediction_classifier_test, df_test_classifier_y)
        axes.set_title('Test sample')
        pdf.savefig(axes.get_figure())

        axes = plot_roc_curve(prediction_classifier_test, df_test_classifier_y)
        axes.set_title('Test sample')
        pdf.savefig(axes.get_figure())

        axes = plot_roc_curve_energy(df_test_classifier, df_test_classifier_y, prediction_classifier_test)
        axes.set_title('Test sample')
        pdf.savefig(axes.get_figure())

        axes = plot_roc_curve(prediction_classifier_train, df_train_classifier_y)
        axes.set_title('Train sample')
        pdf.savefig(axes.get_figure())

        axes = plot_roc_curve_energy(df_train_classifier, df_train_classifier_y, prediction_classifier_train)
        axes.set_title('Train sample')
        pdf.savefig(axes.get_figure())

        axes = plot_2d_source(df_gamma_quality)
        axes.set_title('On-axis $\gamma$')
        pdf.savefig(axes.get_figure())

        axes = plot_2d_source(df_gamma_diffuse_quality)
        axes.set_title('Diffuse $\gamma$')
        pdf.savefig(axes.get_figure())

        axes = plot_2d_source(df_proton_quality)
        axes.set_title('Diffuse $p$')
        pdf.savefig(axes.get_figure())

        axes = plot_2d_disp(df_gamma_quality)
        axes.set_title('On-axis $\gamma$')
        pdf.savefig(axes.get_figure())

        axes = plot_2d_disp(df_gamma_diffuse_quality)
        axes.set_title('Diffuse $\gamma$')
        pdf.savefig(axes.get_figure())

        axes = plot_2d_disp(df_proton_quality)
        axes.set_title('Diffuse $p$')
        pdf.savefig(axes.get_figure())

        fig = plt.figure()
        axes = fig.add_subplot(111)
        bins = np.linspace(
            min(df_gamma_quality['disp_x'].min(), df_gamma_diffuse_quality['disp_x'].min()),
            max(df_gamma_quality['disp_x'].max(), df_gamma_diffuse_quality['disp_x'].max()),
            num=100)
        axes.hist(df_gamma_quality['disp_x'], bins=bins, lw=3, histtype='step',
                  label='Test $\gamma$')
        axes.hist(df_gamma_diffuse_quality['disp_x'], bins=bins, lw=3, histtype='step',
                  label='Train $\gamma$')
        axes.set_xlabel('$DISP_x$ [mm]')
        axes.legend(loc='best')
        pdf.savefig(fig)

        fig = plt.figure()
        axes = fig.add_subplot(111)
        bins = np.linspace(
            min(df_gamma_quality['disp_y'].min(), df_gamma_diffuse_quality['disp_y'].min()),
            max(df_gamma_quality['disp_y'].max(), df_gamma_diffuse_quality['disp_y'].max()),
            num=100)
        axes.hist(df_gamma_quality['disp_y'], bins=bins, lw=3, histtype='step',
                  label='Test $\gamma$')
        axes.hist(df_gamma_diffuse_quality['disp_y'], bins=bins, lw=3,
                  histtype='step', label='Train $\gamma$')

        axes.set_xlabel('$DISP_y$ [mm]')
        axes.legend(loc='best')
        pdf.savefig(fig)

        fig = plt.figure()
        axes = fig.add_subplot(111)
        bins = np.linspace(
            min(df_gamma_quality['disp_r'].min(), df_gamma_diffuse_quality['disp_r'].min()),
            max(df_gamma_quality['disp_r'].max(), df_gamma_diffuse_quality['disp_r'].max()),
            num=100)
        axes.hist(df_gamma_quality['disp_r'], bins=bins, lw=3, histtype='step',
                  label='Test $\gamma$')
        axes.hist(df_gamma_diffuse_quality['disp_r'], bins=bins, lw=3, histtype='step',
                  label='Train $\gamma$')
        axes.set_xlabel('$DISP_r$ [mm]')
        axes.legend(loc='best')
        pdf.savefig(fig)

        fig = plt.figure()
        axes = fig.add_subplot(111)
        bins = np.linspace(
            min(df_gamma_quality['disp_theta'].min(), df_gamma_diffuse_quality['disp_theta'].min()),
            max(df_gamma_quality['disp_theta'].max(), df_gamma_diffuse_quality['disp_theta'].max()),
            num=100)
        axes.hist(df_gamma_quality['disp_theta'], bins=bins, lw=3, histtype='step',
                  label='Test $\gamma$')
        axes.hist(df_gamma_diffuse_quality['disp_theta'], bins=bins, lw=3, histtype='step',
                  label='Train $\gamma$')
        axes.set_xlabel(r'$DISP_{\theta}$ [rad]')
        axes.legend(loc='best')
        pdf.savefig(fig)

        axes = plot_delta_disp(df_train_disp, prediction_disp_train)
        axes.set_title('Train sample')
        pdf.savefig(axes.get_figure())

        axes = plot_delta_disp(df_test_disp, prediction_disp_test)
        axes.set_title('Test sample')
        pdf.savefig(axes.get_figure())

        axes = plot_r68(df_train_disp, prediction_disp_train, energy=df_train_energy['true_energy'])
        axes.set_title('Train Sample')
        pdf.savefig(axes.get_figure())


        axes = plot_r68(df_test_disp, prediction_disp_test, energy=df_test_energy['true_energy'])
        axes.set_title('Test Sample')
        pdf.savefig(axes.get_figure())

        axes = plot_delta_energy(df_train_energy_y, prediction_energy_train)
        axes.set_title('Train Sample')
        pdf.savefig(axes.get_figure())
        axes = plot_delta_energy(df_train_energy_y, prediction_energy_train, kind='reco')
        axes.set_title('Train Sample')
        pdf.savefig(axes.get_figure())

        axes = plot_delta_energy(df_test_energy_y, prediction_energy_test)
        axes.set_title('Test Sample')
        pdf.savefig(axes.get_figure())
        axes = plot_delta_energy(df_test_energy_y, prediction_energy_test, kind='reco')
        axes.set_title('Test Sample')
        pdf.savefig(axes.get_figure())

        axes = plot_resolution_energy(df_train_energy, prediction_energy_train)
        axes.set_title('Train Sample')
        pdf.savefig(axes.get_figure())
        axes = plot_resolution_energy(df_train_energy, prediction_energy_train, kind='reco')
        axes.set_title('Train Sample')
        pdf.savefig(axes.get_figure())

        axes = plot_resolution_energy(df_test_energy, prediction_energy_test)
        axes.set_title('Test Sample')
        pdf.savefig(axes.get_figure())
        axes = plot_resolution_energy(df_test_energy, prediction_energy_test, kind='reco')
        axes.set_title('Test Sample')
        pdf.savefig(axes.get_figure())



        fig = plt.figure()
        axes = fig.add_subplot(111)
        bins = np.linspace(0, np.pi / 2, num=100)
        resolution = df_gamma_quality['alpha'].quantile(0.68)
        df_gamma_quality.hist('alpha', bins=bins, ax=axes, histtype='step',
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
        df_gamma_diffuse_quality.hist('alpha', bins=bins, ax=axes, histtype='step',
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
        resolution = df_gamma_quality['alpha'].quantile(0.68)
        df_gamma_quality.hist('alpha', bins=bins, ax=axes, histtype='step',
                              label='$\gamma$', density=True, lw=3)
        df_proton_quality.hist('alpha', bins=bins, ax=axes, histtype='step',
                               label='Diffuse $p$', density=True, lw=3)
        df_gamma_diffuse_quality.hist('alpha', bins=bins, ax=axes, histtype='step',
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



    print('Figure saved to {}'.format(figure_path))
    # with open(output_filename, 'wb') as f:
    #     pk.dump(rf_classifier, f)
