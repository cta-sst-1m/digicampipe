from digicampipe.io.dl1 import create_data_frame
import glob
import os
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from digicampipe.pointing.disp import angular_distance, cal_cam_source_pos

from digicampipe.visualization.machine_learning import plot_3d_histo, LABELS
import h5py
import argparse

def combine_rate_files(files):

    i = 0
    for file in tqdm(files):

        data = np.load(file)
        if i == 0:
            event_id = data['event_id']
            energy = data['energy']
            x_core = data['x_core']
            y_core = data['y_core']
            h_first_int = data['h_first_int']
            # tel_pos_all = data['tel_pos_all']
            alt = np.deg2rad(90 - data['theta'])
            az = np.deg2rad(data['phi'])

        else:
            event_id = np.append(event_id, data['event_id'])
            energy = np.append(energy, data['energy'])
            x_core = np.append(x_core, data['x_core'])
            y_core = np.append(y_core, data['y_core'])
            h_first_int = np.append(h_first_int, data['h_first_int'])
            alt = np.append(alt, np.deg2rad(90 - data['theta']))
            az = np.append(az, np.deg2rad(data['phi']))

        i += 1

    data = {'event_id': event_id, 'true_energy': energy, 'core_x': x_core,
            'core_y': y_core, 'h_first_int': h_first_int, 'alt': alt, 'az': az,
            }
    data = pd.DataFrame(data)
    return data


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


test_size = 0.2
max_files = None
parser = argparse.ArgumentParser(description='Process some value.')
parser.add_argument('--proton',
                        default='/sst1m/analyzed/mpeshowerfit/tmp/zenith_20_proton_v15.hdf5',
                        type=str, help='Processing version')
parser.add_argument('--gamma',
                        default='/sst1m/analyzed/mpeshowerfit/tmp/zenith_20_gamma_v15.hdf5',
                        type=str, help='Processing version')
parser.add_argument('--gamma_diffuse',
                        default='/sst1m/analyzed/mpeshowerfit/tmp/zenith_20_diffuse_gamma_v15.hdf5',
                        type=str, help='Processing version')
parser.add_argument('--output', default='./',
                        type=str, help='Ouput directory')
args = parser.parse_args()


version = 'v16'
gamma_files = glob.glob(str(args.gamma))
proton_files = glob.glob(args.proton)
gamma_diffuse_files = glob.glob(args.gamma_diffuse)
output_dir = args.output
# output_dir = '/data/sst-1m_prod4b/'

energy_bins = np.linspace(-1, np.log10(500), num=100)
impact_bins = np.linspace(0, 1E3, num=80)
theta_bins = np.linspace(0, np.radians(10), num=40)
tel_alt, tel_az = np.radians(70), np.pi / 2
bins = [energy_bins, impact_bins, theta_bins]

input_dir = '/sst1m/analyzed/mpeshowerfit/'
gamma_basename = 'zenith_20_gamma'
gamma_diffuse_basename = 'zenith_20_diffuse_gamma'
proton_basename = 'zenith_20_proton'
gamma_rate_files = glob.glob(os.path.join(input_dir, gamma_basename + '*_trigger.npz'))
proton_rate_files = glob.glob(os.path.join(input_dir, proton_basename + '*_trigger.npz'))
gamma_diffuse_rate_files = glob.glob(os.path.join(input_dir, gamma_diffuse_basename + '*_trigger.npz'))

gamma_file = os.path.join(output_dir,  'gamma.hdf5')
proton_file = os.path.join(output_dir, 'proton.hdf5')
gamma_diffuse_file = os.path.join(output_dir, 'gamma_diffuse.hdf5')

df_gamma = create_data_frame(gamma_files, max_files=max_files)
df_gamma.to_hdf(gamma_file, key='data', mode='w')
counts_gamma, bins = make_2d_histogram(gamma_rate_files,
                                       energy_bins=energy_bins,
                                       impact_bins=impact_bins,
                                       theta_bins=theta_bins,
                                       tel_alt=tel_alt,
                                       tel_az=tel_az)

with h5py.File(gamma_file, 'a') as h:
    group = h.create_group("trigger")
    group.create_dataset('histo', data=counts_gamma)
    group.create_dataset('bins_energy', data=energy_bins)
    group.create_dataset('bins_impact', data=impact_bins)
    group.create_dataset('bins_theta', data=theta_bins)


df_proton = create_data_frame(proton_files, max_files=max_files)
df_proton.to_hdf(proton_file, key='data', mode='w')
counts_proton, bins = make_2d_histogram(proton_rate_files,
                                        energy_bins=energy_bins,
                                       impact_bins=impact_bins,
                                       theta_bins=theta_bins,
                                       tel_alt=tel_alt,
                                       tel_az=tel_az)
with h5py.File(proton_file, 'a') as h:
    group = h.create_group("trigger")
    group.create_dataset('histo', data=counts_proton)
    group.create_dataset('bins_energy', data=energy_bins)
    group.create_dataset('bins_impact', data=impact_bins)
    group.create_dataset('bins_theta', data=theta_bins)



df_gamma_diffuse = create_data_frame(gamma_diffuse_files, max_files=max_files)
df_gamma_diffuse.to_hdf(gamma_diffuse_file, key='data', mode='w')
counts_gamma_diffuse, bins = make_2d_histogram(gamma_diffuse_rate_files,
                                               energy_bins=energy_bins,
                                               impact_bins=impact_bins,
                                               theta_bins=theta_bins,
                                               tel_alt=tel_alt,
                                               tel_az=tel_az
                                               )

with h5py.File(gamma_diffuse_file, 'a') as h:
    group = h.create_group("trigger")
    group.create_dataset('histo', data=counts_gamma_diffuse)
    group.create_dataset('bins_energy', data=energy_bins)
    group.create_dataset('bins_impact', data=impact_bins)
    group.create_dataset('bins_theta', data=theta_bins)


with h5py.File(gamma_file, 'r') as h:
    counts_gamma = np.array(h['trigger']['histo'])

with h5py.File(proton_file, 'r') as h:
    counts_proton = np.array(h['trigger']['histo'])

with h5py.File(gamma_diffuse_file, 'r') as h:
    counts_gamma_diffuse = np.array(h['trigger']['histo'])

with PdfPages(os.path.join(output_dir, 'figure_feature_distribution_' + version +'.pdf')) as pdf:

    axes = plot_3d_histo(counts_gamma, bins)
    axes.set_xlabel(LABELS['log_true_energy'])
    axes.set_ylabel(LABELS['impact'])
    axes.set_title('On-axis $\gamma$')
    pdf.savefig(axes.get_figure())

    axes = plot_3d_histo(counts_gamma, bins=bins, axis=0)
    axes.set_xlabel(LABELS['impact'])
    axes.set_ylabel(LABELS['angular_distance'])
    axes.set_title('On-axis $\gamma$')

    pdf.savefig(axes.get_figure())

    axes = plot_3d_histo(counts_gamma, bins, axis=1)
    axes.set_xlabel(LABELS['log_true_energy'])
    axes.set_ylabel(LABELS['angular_distance'])
    axes.set_title('On-axis $\gamma$')

    pdf.savefig(axes.get_figure())

    axes = plot_3d_histo(counts_proton, bins)
    axes.set_xlabel(LABELS['log_true_energy'])
    axes.set_ylabel(LABELS['impact'])
    axes.set_title('Diffuse $p$')
    pdf.savefig(axes.get_figure())

    axes = plot_3d_histo(counts_proton, bins=bins, axis=0)
    axes.set_xlabel(LABELS['impact'])
    axes.set_ylabel(LABELS['angular_distance'])
    axes.set_title('Diffuse $p$')
    pdf.savefig(axes.get_figure())

    axes = plot_3d_histo(counts_proton, bins, axis=1)
    axes.set_xlabel(LABELS['log_true_energy'])
    axes.set_ylabel(LABELS['angular_distance'])
    axes.set_title('Diffuse $p$')
    pdf.savefig(axes.get_figure())

    axes = plot_3d_histo(counts_gamma_diffuse, bins)
    axes.set_xlabel(LABELS['log_true_energy'])
    axes.set_ylabel(LABELS['impact'])
    axes.set_title('Diffuse $\gamma$')
    pdf.savefig(axes.get_figure())

    axes = plot_3d_histo(counts_gamma_diffuse, bins=bins, axis=0)
    axes.set_xlabel(LABELS['impact'])
    axes.set_ylabel(LABELS['angular_distance'])
    axes.set_title('Diffuse $\gamma$')
    pdf.savefig(axes.get_figure())

    axes = plot_3d_histo(counts_gamma_diffuse, bins, axis=1)
    axes.set_xlabel(LABELS['log_true_energy'])
    axes.set_ylabel(LABELS['angular_distance'])
    axes.set_title('Diffuse $\gamma$')
    pdf.savefig(axes.get_figure())
    features = df_gamma.columns

    for feature in features:

        if feature == 'valid':
            continue

        x, y, z = df_gamma[feature], df_proton[feature], df_gamma_diffuse[feature]
        bins = np.linspace(min(np.nanmin(x), np.nanmin(y), np.nanmin(z)), max(np.nanmax(x), np.nanmax(y), np.nanmax(z)), num=100)
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
