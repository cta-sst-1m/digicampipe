import glob
import os
import pandas as pd
import numpy as np
from tqdm import tqdm

from digicampipe.pointing.disp import angular_distance, cal_cam_source_pos

import h5py
import argparse


def make_2d_histogram(files, energy_bins=np.linspace(-1, np.log10(500), num=100),
                      impact_bins=np.linspace(0, 2E3, num=80),
                      theta_bins=np.linspace(0, np.radians(7), num=40),
                      tel_alt=np.radians(70), tel_az=np.pi/2):

    bins = [energy_bins, impact_bins, theta_bins]
    shape = (len(energy_bins) - 1, len(impact_bins) - 1, len(theta_bins) - 1)
    counts = np.zeros(shape=shape)

    i = 0
    for file in tqdm(files):

        data = np.load(file)
        energy = np.log10(data['energy'])
        x_core = data['x_core']
        y_core = data['y_core']
        r = np.sqrt(x_core**2 + y_core**2)
        alt = np.deg2rad(90 - data['theta'])
        az = np.deg2rad(data['phi'])
        delta = angular_distance(alt, az, tel_alt, tel_az)
        data, _ = np.histogramdd([energy, r, delta], bins=bins)
        counts += data

        i += 1
    return counts, bins


parser = argparse.ArgumentParser(description='Process some value.')
parser.add_argument('--input', default='./', type=str, help='Input directory')
parser.add_argument('--output', default='./', type=str, help='Ouput directory')
args = parser.parse_args()

e_min, e_max = 0.1, 500
r_max = 2E3
delta_max = np.radians(7)
tel_alt, tel_az = np.radians(70), np.pi / 2

energy_bins = np.linspace(np.log10(e_min), np.log10(e_max), num=100)
impact_bins = np.linspace(0, r_max, num=80)
theta_bins = np.linspace(0, delta_max, num=40)
bins = [energy_bins, impact_bins, theta_bins]

input_dir = args.input
output_dir = args.output
gamma_basename = 'zenith_20_gamma'
gamma_diffuse_basename = 'zenith_20_diffuse_gamma'
proton_basename = 'zenith_20_proton'
gamma_rate_files = glob.glob(os.path.join(input_dir, gamma_basename + '*_trigger.npz'))
proton_rate_files = glob.glob(os.path.join(input_dir, proton_basename + '*_trigger.npz'))
gamma_diffuse_rate_files = glob.glob(os.path.join(input_dir, gamma_diffuse_basename + '*_trigger.npz'))

gamma_file = os.path.join(output_dir, gamma_basename + '.h5')
proton_file = os.path.join(output_dir, proton_basename + '.h5')
gamma_diffuse_file = os.path.join(output_dir, gamma_diffuse_basename + '.h5')

if os.path.exists(gamma_file) or os.path.exists(gamma_diffuse_file) or os.path.exists(proton_file):

    raise FileExistsError('One of the output file already exists')

counts_gamma, bins = make_2d_histogram(gamma_rate_files,
                                       energy_bins=energy_bins,
                                       impact_bins=impact_bins,
                                       theta_bins=theta_bins,
                                       tel_alt=tel_alt,
                                       tel_az=tel_az)

with h5py.File(gamma_file, 'w') as h:
    group = h.create_group("trigger")
    group.create_dataset('histo', data=counts_gamma)
    group.create_dataset('bins_energy', data=energy_bins)
    group.create_dataset('bins_impact', data=impact_bins)
    group.create_dataset('bins_theta', data=theta_bins)

counts_proton, bins = make_2d_histogram(proton_rate_files,
                                        energy_bins=energy_bins,
                                       impact_bins=impact_bins,
                                       theta_bins=theta_bins,
                                       tel_alt=tel_alt,
                                       tel_az=tel_az)

with h5py.File(proton_file, 'w') as h:
    group = h.create_group("trigger")
    group.create_dataset('histo', data=counts_proton)
    group.create_dataset('bins_energy', data=energy_bins)
    group.create_dataset('bins_impact', data=impact_bins)
    group.create_dataset('bins_theta', data=theta_bins)

counts_gamma_diffuse, bins = make_2d_histogram(gamma_diffuse_rate_files,
                                               energy_bins=energy_bins,
                                               impact_bins=impact_bins,
                                               theta_bins=theta_bins,
                                               tel_alt=tel_alt,
                                               tel_az=tel_az
                                               )

with h5py.File(gamma_diffuse_file, 'w') as h:
    group = h.create_group("trigger")
    group.create_dataset('histo', data=counts_gamma_diffuse)
    group.create_dataset('bins_energy', data=energy_bins)
    group.create_dataset('bins_impact', data=impact_bins)
    group.create_dataset('bins_theta', data=theta_bins)
