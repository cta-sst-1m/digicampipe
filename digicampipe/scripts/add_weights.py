import numpy as np
from digicampipe.io.dl1 import read_hdf5
from digicampipe.scripts.all_in_one_reco import get_data
from digicampipe.visualization.machine_learning import get_rate_gamma, get_rate_proton
import argparse
from tqdm import tqdm
import h5py
import pandas as pd

max_events = None

parser = argparse.ArgumentParser(description='Process some value.')
parser.add_argument('--proton',
                    default=None,
                    type=str, help='Processing version')
parser.add_argument('--gamma',
                    default=None,
                    type=str, help='Processing version')

parser.add_argument('--gamma_rate',
                    default=None,
                    type=str, help='Processing version')


parser.add_argument('--proton_rate',
                    default=None,
                    type=str, help='Processing version')

args = parser.parse_args()
gamma_file = args.gamma
proton_file = args.proton
gamma_rates_file = args.gamma_rate
proton_rates_file = args.proton_rate

def get_count_histogram(file):

    with h5py.File(file, 'r') as f:

        count = np.array(f['trigger/histo'])
        bins_energy = np.array(f['trigger/bins_energy'])
        bins_impact = np.array(f['trigger/bins_impact'])
        bins_delta = np.array(f['trigger/bins_theta'])

    bins = [bins_energy, bins_impact, bins_delta]
    return count, bins


def add_weights(df, count, bins, kind):

    bins_energy, bins_impact, bins_delta = bins
    df['weights'] = np.ones(len(df)) * np.nan

    if kind == 'proton':

        rate = get_rate_proton(bins_energy=bins_energy,
                               bins_impact=bins_impact,
                               bins_theta=bins_delta)

    elif kind == 'gamma':

        rate = get_rate_gamma(bins_energy=bins_energy, bins_impact=bins_impact)
        print(count.shape)
        print(rate.shape)
        print(bins_energy.shape, bins_impact.shape, bins_delta.shape)
        count = count.sum(axis=-1)

    weights = rate / count
    mask = np.isfinite(weights)
    weights[~mask] = 0.0

    for i in tqdm(range(len(bins_energy) - 1)):
        for j in range(len(bins_impact) - 1):

            if kind == 'proton':
                for k in range(len(bins_delta) - 1):
                    mask = (np.log10(df['true_energy']) > bins_energy[i]) & \
                           (np.log10(df['true_energy']) <= bins_energy[i + 1]) & \
                           (df['impact'] > bins_impact[j]) & \
                           (df['impact'] <= bins_impact[j + 1]) & \
                           (df['delta'] > bins_delta[k]) & \
                           (df['delta'] <= bins_delta[k + 1])

                    df.loc[mask, 'weights']  = weights[i, j, k]
            elif kind == 'gamma':
                mask = (np.log10(df['true_energy']) > bins_energy[i]) & \
                       (np.log10(df['true_energy']) <= bins_energy[i + 1]) & \
                       (df['impact'] > bins_impact[j]) & \
                       (df['impact'] <= bins_impact[j + 1])
                df.loc[mask, 'weights'] = weights[i, j]
    return df

count_gamma, bins_gamma = get_count_histogram(gamma_rates_file)
count_proton, bins_proton = get_count_histogram(proton_rates_file)


def write_trigger_info(file, count, bins):

    with h5py.File(file, 'r+') as h:
        group = h.create_group("trigger")
        group.create_dataset('histo', data=count)
        group.create_dataset('bins_energy', data=bins[0])
        group.create_dataset('bins_impact', data=bins[1])
        group.create_dataset('bins_theta', data=bins[2])

df_gamma = pd.read_hdf(gamma_file, key='data')
print(df_gamma['weights'].sum())

df_gamma = add_weights(df_gamma, count_gamma, bins=bins_gamma, kind='gamma')
df_gamma.to_hdf()



print(count_gamma.sum(), count_proton.sum())
