
import numpy as np
import matplotlib.pyplot as plt
from digicampipe.visualization.machine_learning import plot_features, plot_init_vs_fit,plot_2d_histogram
import os

from digicampipe.io.dl1 import read_hdf5, combine_datasets
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd
from digicampipe.pointing.disp import cal_cam_source_pos
from tqdm import tqdm
import argparse

FOCAL = 5600

def compute_source_xy(df):
    alt, az = np.array(df['alt']), np.array(df['az'])
    tel_alt, tel_az = np.array(df['tel_alt']), np.array(df['tel_az'])

    x = np.zeros(len(df))
    y = np.zeros(len(df))

    for i in tqdm(range(len(df))):
        x[i], y[i] = cal_cam_source_pos(alt[i], az[i],
                                        tel_alt[i], tel_az[i],
                                        focal_length=FOCAL)

    df['source_x'], df['source_y'] = x, y
    df['disp_x'] = df['source_x'] - df['x']
    df['disp_y'] = df['source_y'] - df['y']
    df['disp_r'] = np.sqrt(df['disp_x']**2 + df['disp_y']**2)
    df['disp_theta'] = np.arctan(df['disp_y']/df['disp_x'])
    df['velocity'] = 1 / df['slope']
    return df


parser = argparse.ArgumentParser(description='Process some value.')
parser.add_argument('--max_events', default=None, type=int,
                    help='sum the integers (default: find the max)')
parser.add_argument('--proton',  default='/sst1m/analyzed/mpeshowerfit/zenith_20_proton_v15.hdf5',
                    type=str, help='Processing version')
parser.add_argument('--gamma', default='/sst1m/analyzed/mpeshowerfit/zenith_20_gamma_v15.hdf5',
                    type=str, help='Processing version')
parser.add_argument('--gamma_diffuse',default='/sst1m/analyzed/mpeshowerfit/zenith_20_diffuse_gamma_v15.hdf5',
                    type=str, help='Processing version')
parser.add_argument('--output_directory', default='.',
                    type=str, help='Processing version')
args = parser.parse_args()

max_events = args.max_events # None # 1000 # 10000
output_dir = args.output_directory

gamma_files = args.gamma
proton_files = args.proton
gamma_diffuse_files = args.gamma_diffuse

df_gamma = read_hdf5(gamma_files, max_events=max_events)
print("Gamma sample loaded with {} rows".format(len(df_gamma)))
df_gamma = compute_source_xy(df_gamma)
df_proton = read_hdf5(proton_files, max_events=max_events)
print("Proton sample loaded with {} rows".format(len(df_proton)))
df_proton = compute_source_xy(df_proton)
df_gamma_diffuse = read_hdf5(gamma_diffuse_files, max_events=max_events)
print("Gamma diffuse sample loaded with {} rows".format(len(df_gamma_diffuse)))
df_gamma_diffuse = compute_source_xy(df_gamma_diffuse)

figure_path = os.path.join(output_dir, 'figure_parameters_v15.pdf')


fit_params = ['alpha', 'intensity', 'length', 'width', 'phi', 'psi',
              'r', 'x', 'y', 'slope', 'intercept']

with PdfPages(figure_path) as pdf:

    labels = ['On-axis $\gamma$', 'Diffuse $\gamma$', 'Diffuse $p$']
    for i, df in enumerate([df_gamma, df_gamma_diffuse, df_proton]):
        for feature in fit_params:
            axes = plot_init_vs_fit(df, feature, bins=int(min(200, np.sqrt(len(df)))))
            axes.set_title(labels[i])
            pdf.savefig(axes.get_figure())

        axes = plot_2d_histogram(df, 'true_energy', 'area', log=(True, False))
        axes.set_title(labels[i])
        pdf.savefig(axes.get_figure())

        limits_slope = limits=((-np.inf, np.inf), (-0.5, 0.5))
        bins_slope = (100, 2000)

        axes = plot_2d_histogram(df, 'true_energy', 'slope', log=(True, False), bins=bins_slope, limits=limits_slope)
        axes.set_title(labels[i])
        pdf.savefig(axes.get_figure())


        axes = plot_2d_histogram(df, 'true_energy', 'density_w',log=(True, False))
        axes.set_title(labels[i])
        pdf.savefig(axes.get_figure())

        axes = plot_2d_histogram(df, 'true_energy', 'length',log=(True, False))
        axes.set_title(labels[i])
        pdf.savefig(axes.get_figure())

        axes = plot_2d_histogram(df, 'true_energy', 'intensity',log=(True, True))
        axes.set_title(labels[i])
        pdf.savefig(axes.get_figure())

        axes = plot_2d_histogram(df, 'disp_theta', 'psi',)
        axes.set_title(labels[i])
        pdf.savefig(axes.get_figure())

        axes = plot_2d_histogram(df, 'disp_r', 'length',)
        axes.set_title(labels[i])
        pdf.savefig(axes.get_figure())

        axes = plot_2d_histogram(df, 'disp_r', 'slope', bins=bins_slope, limits=limits_slope)
        axes.set_title(labels[i])
        pdf.savefig(axes.get_figure())

        axes = plot_2d_histogram(df, 'disp_r', 'skewness_l', )
        axes.set_title(labels[i])
        pdf.savefig(axes.get_figure())

    for feature in df_gamma.columns:
        try:
            axes = plot_features(feature, df_gamma=df_gamma, df_proton=df_proton, df_gamma_diffuse=df_gamma_diffuse)
            pdf.savefig(axes.get_figure())
        except KeyError:
            print('Could not find {}'.format(feature))
            pass



print("Figure saved to {}".format(figure_path))