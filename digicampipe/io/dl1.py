import pandas as pd
import numpy as np
from sklearn.utils import resample
from tqdm import tqdm
from digicampipe.image.disp import compute_leakage
import random
import h5py
from digicampipe.pointing.disp import angular_distance, cal_cam_source_pos
from digicampipe.image.hillas import compute_alpha


def create_data_frame(files, max_files=None, FOCAL=5600):

    found_file = False

    while not found_file:

        try:
            df_0 = pd.read_hdf(files[0], key='data/hillas')
            df_1 = pd.read_hdf(files[0], key='data/mc')
            df_2 = pd.read_hdf(files[0], key='data/timing')
            df_3 = pd.read_hdf(files[0], key='data/init_timing')
            df_3.rename(columns=lambda x: x + '_init', inplace=True)
            df_4 = pd.read_hdf(files[0], key='data/init_hillas')
            df_4.rename(columns=lambda x: x + '_init', inplace=True)
        except KeyError:

            random.shuffle(files)
        else:
            found_file = True
            print('First file found')

    # print(len(df))
    df = pd.concat([df_0, df_1, df_2, df_3, df_4], axis=1, sort=False)
    # print(df.columns)
    # print(len(df))

    if max_files is None:
        n_files = len(files)
    else:
        n_files = min(max_files, len(files))

    for i in tqdm(range(1, n_files)):

        try:

            df_0 = pd.read_hdf(files[i], key='data/hillas')
            df_1 = pd.read_hdf(files[i], key='data/mc')
            df_2 = pd.read_hdf(files[i], key='data/timing')
            df_3 = pd.read_hdf(files[i], key='data/init_timing')
            df_3.rename(columns=lambda x: x + '_init', inplace=True)
            df_4 = pd.read_hdf(files[i], key='data/init_hillas')
            df_4.rename(columns=lambda x: x + '_init', inplace=True)

            df_merged = pd.concat([df_0, df_1, df_2, df_3, df_4], axis=1, sort=False)
            df = df.append(df_merged)
        except (OSError, KeyError):
            pass

    df['wol'] = df['width'] / df['length']
    is_nan = (df.isna().sum(axis=1) > 0)
    df['valid'] = ~is_nan
    mask = df['particle'] == 101
    index = np.arange(len(df))
    index = index[mask]
    df['particle'].iloc[index] = 1
    df['density'] = df['intensity'] / (df['width'] * df['length'])
    df['density_l'] = df['intensity'] / df['length']
    df['density_w'] = df['intensity'] / df['width']
    df['area'] = df['width'] * df['length'] * np.pi
    df['delta'] = angular_distance(df['alt'], df['az'], df['tel_alt'], df['tel_az'])
    df['impact'] = np.sqrt(df['core_x']**2 + df['core_y']**2)

    alt, az = np.array(df['alt']), np.array(df['az'])
    tel_alt, tel_az = np.array(df['tel_alt']), np.array(df['tel_az'])

    if ('source_x' not in df.columns) or ('source_y' not in df.columns):
        x = np.zeros(len(df))
        y = np.zeros(len(df))
        for i in tqdm(range(len(df))):
            x[i], y[i] = cal_cam_source_pos(alt[i], az[i],
                                            tel_alt[i], tel_az[i],
                                            focal_length=FOCAL)

        df['source_x'], df['source_y'] = x, y
        df['disp_x'] = df['source_x'] - df['x']
        df['disp_y'] = df['source_y'] - df['y']
        df['disp_r'] = np.sqrt(df['disp_x'] ** 2 + df['disp_y'] ** 2)
        df['disp_theta'] = np.arctan2(df['disp_y'], df['disp_x'])
    df['velocity'] = 1. / df['slope']

    mask = df['wol'] > 1

    df.loc[mask, 'length'], df.loc[mask, 'width'] = df.loc[mask, 'width'], \
                                                    df.loc[mask, 'length']
    df.loc[mask, 'wol'] = 1. / df.loc[mask, 'wol']
    df.loc[mask, 'skewness_l'], df.loc[mask, 'skewness_w'] = df.loc[
                                                                 mask, 'skewness_w'], \
                                                             df.loc[
                                                                 mask, 'skewness_l']
    df.loc[mask, 'kurtosis_l'], df.loc[mask, 'kurtosis_w'] = df.loc[
                                                                 mask, 'kurtosis_w'], \
                                                             df.loc[
                                                                 mask, 'kurtosis_l']
    df.loc[mask, 'density_l'], df.loc[mask, 'density_w'] = df.loc[
                                                               mask, 'density_w'], \
                                                           df.loc[
                                                               mask, 'density_l']
    mask_1 = mask & (df['psi'] <= np.pi / 2)
    mask_2 = mask & (df['psi'] > np.pi / 2)
    df.loc[mask_1, 'psi'] = df.loc[mask_1, 'psi'] + np.pi / 2
    df.loc[mask_2, 'psi'] = df.loc[mask_2, 'psi'] - np.pi / 2

    df['t_68'] = np.abs(df['slope'] * df['length'])
    df['alpha'] = compute_alpha(df['phi'], df['psi'])

    if 'leakage' not in df.columns:

        df['leakage'] = df.apply(lambda row: compute_leakage(x=row['x'],
                                                             y=row['y'],
                                                             width=row['width'],
                                                             length=row['length'],
                                                             psi=row['psi'],
                                                             n_sigma=3), axis=1)

    return df


def read_event_histogram(file):

    with h5py.File(file, 'r') as h:

        histo = np.array(h['trigger/histo'])
        bins_energy = np.array(h['trigger/bins_energy'])
        bins_impact = np.array(h['trigger/bins_impact'])
        bins_theta = np.array(h['trigger/bins_theta'])

    return histo, bins_energy, bins_impact, bins_theta


def read_hdf5(file, max_events=None, kind='fit', FOCAL=5600):

    df = pd.read_hdf(file, key='data',)
    df = df.sample(frac=1) # shuffles the data sample

    n_events = len(df)
    factor = 1.0
    if max_events is not None:
        df = df.iloc[:max_events]
        factor = max_events / n_events

    for feature in df.columns:

        if '_err' in feature:
            df = df.drop(feature, axis=1)

    df['delta'] = angular_distance(df['alt'], df['az'], df['tel_alt'], df['tel_az'])
    df['impact'] = np.sqrt(df['core_x'] ** 2 + df['core_y'] ** 2)
    mask = df['particle'] == 101
    index = np.arange(len(df))
    index = index[mask]
    df['particle'].iloc[index] = 1
    is_nan = (df.isna().sum(axis=1) > 0)
    df['valid'] = ~is_nan


    fit_params = ['alpha', 'intensity', 'length', 'width', 'phi', 'psi',
                  'r', 'x', 'y', 'slope', 'intercept', 'kurtosis_l',
                  'kurtosis_w', 'skewness_l', 'skewness_w']

    if kind == 'init':
        for fit_param in fit_params:
            df[fit_param] = df[fit_param + '_init']
        df = df.drop('leakage', axis=1)


    for feature in df.columns:
        if '_init' in feature:
            df = df.drop(feature, axis=1)

    df['wol'] = df['width'] / df['length']

    if kind == 'fit':

        mask = df['wol'] > 1
        df.loc[mask, 'length'], df.loc[mask, 'width'] = df.loc[mask, 'width'], \
                                                        df.loc[mask, 'length']
        df.loc[mask, 'wol'] = 1. / df.loc[mask, 'wol']

        if ('skewness_l' in df.columns) and  ('skewness_w' in df.columns) and ('kurtosis_l' in df.columns) and ('kurtosis_w' in df.columns):
            df.loc[mask, 'skewness_l'], df.loc[mask, 'skewness_w'] = df.loc[
                                                                     mask, 'skewness_w'], \
                                                                 df.loc[
                                                                     mask, 'skewness_l']
            df.loc[mask, 'kurtosis_l'], df.loc[mask, 'kurtosis_w'] = df.loc[

                                                                     mask, 'kurtosis_w'], \
                                                                 df.loc[
                                                                     mask, 'kurtosis_l']

        mask_1 = mask & (df['psi'] <= np.pi / 2)
        mask_2 = mask & (df['psi'] > np.pi / 2)
        df.loc[mask_1, 'psi'] = df.loc[mask_1, 'psi'] + np.pi / 2
        df.loc[mask_2, 'psi'] = df.loc[mask_2, 'psi'] - np.pi / 2

    df['alpha'] = compute_alpha(df['phi'], df['psi'])
    df['area'] = df['width'] * df['length'] * np.pi
    df['t_68'] = np.abs(df['slope'] * df['length'])
    df['density'] = df['intensity'] / (df['area'])
    df['density_l'] = df['intensity'] / df['length']
    df['density_w'] = df['intensity'] / df['width']
    df['log_intensity'] = np.log10(df['intensity'])

    if 'weights' not in df.columns:
        df['weights'] = np.ones(len(df))

    if 'leakage' not in df.columns:
        df['leakage'] = df.apply(lambda row: compute_leakage(x=row['x'],
                                                             y=row['y'],

                                                             width=row['width'],
                                                             length=row[
                                                  'length'],
                                                             psi=row['psi'], n_sigma=3), axis=1)
    if ('source_x' not in df.columns) or ('source_y' not in df.columns):
        x = np.zeros(len(df))
        y = np.zeros(len(df))
        alt, az = np.array(df['alt']), np.array(df['az'])
        tel_alt, tel_az = np.array(df['tel_alt']), np.array(df['tel_az'])
        for i in tqdm(range(len(df))):
            x[i], y[i] = cal_cam_source_pos(alt[i], az[i],
                                            tel_alt[i], tel_az[i],
                                            focal_length=FOCAL)

        df['source_x'], df['source_y'] = x, y

    df['disp_x'] = df['source_x'] - df['x']
    df['disp_y'] = df['source_y'] - df['y']
    df['disp_r'] = np.sqrt(df['disp_x'] ** 2 + df['disp_y'] ** 2)
    df['disp_theta'] = np.arctan2(df['disp_y'], df['disp_x'])
    df['velocity'] = 1. / df['slope']

    return df, factor


def combine_datasets(df_proton, df_gamma, random_state=None):
    df = df_proton.append(df_gamma)

    n_0, n_1 = (df.particle == 0).sum(), (df.particle == 1).sum()
    n_samples = min(n_0, n_1)
    if n_0 > n_1:

        df_majority = df[df.particle == 0]
        df_minority = df[df.particle == 1]

    else:

        df_majority = df[df.particle == 1]
        df_minority = df[df.particle == 0]

    df_majority_downsampled = resample(df_majority,
                                       replace=False,
                                       # sample without replacement
                                       n_samples=n_samples,
                                       # to match minority class
                                       random_state=random_state)  # reproducible results

    # Combine minority class with downsampled majority class
    df_downsampled = pd.concat([df_majority_downsampled, df_minority])

    # Display new class counts
    df_downsampled.particle.value_counts()

    return df_downsampled


def quality_cuts(df, intensity_cut=15):

    mask = (df['intensity'] >= intensity_cut) & (df['log_lh'] >= -6) & (df['leakage'] >= 0.75) & (np.abs(df['slope']) < 0.5)
    mask = mask & (df['wol'] < 1) & (df['wol'] > 0.1)
    mask = np.arange(len(df))[mask]
    df = df.iloc[mask]
    with pd.option_context('mode.use_inf_as_null', True): # Drops infinite values
        df = df.dropna()

    return df


def remove_error_params(df):

    features = df.columns
    dropped_features = []

    for feature in features:

        if '_err' in feature:

            dropped_features.append(feature)

    df = df.drop(dropped_features, axis=1)

    return df
