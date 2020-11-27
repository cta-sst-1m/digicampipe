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
import astropy.units as u
from digicampipe.pointing.disp import sky_to_camera, cal_cam_source_pos
from tqdm import tqdm

FOCAL = 5.6*1E3


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
    axes.legend(loc='best')
    plt.colorbar(H[3], label='count []')

    return axes


def compute_source_xy(df):
    alt, az = np.array(df['alt']), np.array(df['az'])
    tel_alt, tel_az = np.array(df['tel_alt']), np.array(df['tel_az'])
    # pos = sky_to_camera(alt=alt * u.rad, az=az * u.rad, focal=5.6*1000 * u.mm, pointing_alt=tel_alt * u.rad, pointing_az=tel_az * u.rad, )

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
    return df


test_size = 0.25
max_events = None
n_estimators = 20
# input_dir = '/data/sst-1m_prod4b/'
# input_dir = '/work/se2/alispach/sst-1m_prod4b/'
input_dir = '/sst1m/analyzed/mpeshowerfit/'
# output_dir = '/data/sst-1m_prod4b/'
output_dir = '/work/se2/alispach/sst-1m_prod4b/'
output_dir = '/sst1m/analyzed/mpeshowerfit/'

#gamma_diffuse_file = os.path.join(input_dir, 'gamma_diffuse/gamma_diffuse_v2.hdf5')
gamma_diffuse_file = os.path.join(input_dir, 'gamma_v11.hdf5')
gamma_file = os.path.join(input_dir, 'gamma_v11.hdf5')
# gamma_file = os.path.join(input_dir, 'gamma/gamma_v2.hdf5')
df_gamma_diffuse = read_hdf5(gamma_diffuse_file, max_events=max_events)
df_gamma = read_hdf5(gamma_file, max_events=max_events)
df_gamma_diffuse = quality_cuts(df_gamma_diffuse, intensity_cut=100)
df_gamma = quality_cuts(df_gamma, intensity_cut=100)

df_gamma_diffuse = compute_source_xy(df_gamma_diffuse)
df_gamma = compute_source_xy(df_gamma)

df_gamma = df_gamma.dropna()
df_gamma_diffuse = df_gamma_diffuse.dropna()

df_gamma_diffuse_train, df_gamma_diffuse_test = train_test_split(df_gamma_diffuse, test_size=test_size)
df_gamma_train, df_gamma_test = train_test_split(df_gamma, test_size=test_size, )


# print("Test sample contains {:d} events\n".format(len(df_test)))
# print("Train sample contains {:d} events\n".format(len(df_train)))

dropped_features = ['true_energy', 'event_id', 'particle',
                    'tel_id', 'valid', 'intercept', 'alt', 'az',
                    'tel_alt', 'tel_az', 'source_x', 'source_y',
                    'disp_x', 'disp_y', 'disp_r', 'disp_theta',
                    'phi',
                    'r', 'x', 'y', 'alpha', 'r_err', 'x_err', 'y_err', 'alpha_err',]

target = ['disp_x', 'disp_y']
# target = ['source_x', 'source_y']

df_gamma_diffuse_train_X, df_gamma_diffuse_train_y = df_gamma_diffuse_train.drop(dropped_features, axis=1), df_gamma_diffuse_train[target]
df_gamma_train_X, df_gamma_train_y = df_gamma_train.drop(dropped_features, axis=1), df_gamma_train[target]
df_gamma_diffuse_test_X, df_gamma_diffuse_test_y = df_gamma_diffuse_test.drop(dropped_features, axis=1), df_gamma_diffuse_test[target]
df_gamma_test_X, df_gamma_test_y = df_gamma_test.drop(dropped_features, axis=1), df_gamma_test[target]
features = df_gamma_diffuse_train_X.columns

print("Training direction regressor with\n {}".format(features))

rf_direction = RandomForestRegressor(n_estimators=n_estimators, n_jobs=-1, verbose=3)
# rf_direction.fit(df_gamma_diffuse_train_X, df_gamma_diffuse_train_y)
rf_direction.fit(df_gamma_train_X, df_gamma_train_y)

reco_direction_gamma_diffuse_test = rf_direction.predict(df_gamma_diffuse_test_X)
reco_direction_gamma_test = rf_direction.predict(df_gamma_test_X)
reco_direction_gamma_diffuse_train = rf_direction.predict(df_gamma_diffuse_train_X)
reco_direction_gamma_train = rf_direction.predict(df_gamma_train_X)
importance = rf_direction.feature_importances_
r2_gamma_diffuse_test = r2_score(df_gamma_diffuse_test_y, reco_direction_gamma_diffuse_test)
r2_gamma_test = r2_score(df_gamma_test_y, reco_direction_gamma_test)
r2_gamma_diffuse_train = r2_score(df_gamma_diffuse_train_y, reco_direction_gamma_diffuse_train)
r2_gamma_train = r2_score(df_gamma_train_y, reco_direction_gamma_train)

print('Train Gamma diffuse R^2 score {:.6f}'.format(r2_gamma_diffuse_train))
print('Train Gamma R^2 score {:.6f}'.format(r2_gamma_train))
print('Test Gamma diffuse R^2 score {:.6f}'.format(r2_gamma_diffuse_test))
print('Test Gamma R^2 score {:.6f}'.format(r2_gamma_test))

n_x, n_y = 300, 300

with PdfPages(os.path.join(output_dir,'figure_direction_regressor.pdf')) as pdf:

    fig = plt.figure()
    axes = fig.add_subplot(111)
    axes.hist2d(df_gamma['source_x'],
                df_gamma['source_y'],
                bins=100,
                label='$\gamma$')
    axes.set_xlabel('$x_{source}$ [mm]')
    axes.set_ylabel('$y_{source}$ [mm]')
    axes.legend(loc='best')
    pdf.savefig(fig)

    fig = plt.figure()
    axes = fig.add_subplot(111)
    axes.hist2d(df_gamma_diffuse['source_x'],
                df_gamma_diffuse['source_y'],
                bins=100,
                label='Diffuse $\gamma$')
    axes.set_xlabel('$x_{source}$ [mm]')
    axes.set_ylabel('$y_{source}$ [mm]')
    axes.legend(loc='best')
    pdf.savefig(fig)


    fig = plt.figure()
    axes = fig.add_subplot(111)
    axes.hist2d(df_gamma['disp_x'],
                 df_gamma['disp_y'],
                 bins=100,
                label='$\gamma$')
    axes.set_xlabel('$DISP_x$ [mm]')
    axes.set_ylabel('$DISP_y$ [mm]')
    axes.legend(loc='best')
    pdf.savefig(fig)

    fig = plt.figure()
    axes = fig.add_subplot(111)
    axes.hist2d(df_gamma_diffuse['disp_x'],
                 df_gamma_diffuse['disp_y'],
                 bins=100,
                label='$Diffuse \gamma$')
    axes.set_xlabel('$DISP_x$ [mm]')
    axes.set_ylabel('$DISP_y$ [mm]')
    axes.legend(loc='best')
    pdf.savefig(fig)


    fig = plt.figure()
    axes = fig.add_subplot(111)
    bins = np.linspace(min(df_gamma['disp_x'].min(), df_gamma_diffuse['disp_x'].min()), max(df_gamma['disp_x'].max(), df_gamma_diffuse['disp_x'].max()), num=100)
    axes.hist(df_gamma_test['disp_x'], bins=bins, lw=3, histtype='step', label='Test $\gamma$')
    axes.hist(df_gamma_diffuse_test['disp_x'], bins=bins, lw=3, histtype='step', label='Test Diffuse $\gamma$')
    axes.hist(df_gamma_train['disp_x'], bins=bins, lw=3, histtype='step', label='Train $\gamma$')
    axes.hist(df_gamma_diffuse_train['disp_x'], bins=bins, lw=3, histtype='step', label='Train Diffuse $\gamma$')
    axes.set_xlabel('$DISP_x$ [mm]')
    axes.legend(loc='best')
    pdf.savefig(fig)

    fig = plt.figure()
    axes = fig.add_subplot(111)
    bins = np.linspace(
        min(df_gamma['disp_y'].min(), df_gamma_diffuse['disp_y'].min()),
        max(df_gamma['disp_y'].max(), df_gamma_diffuse['disp_y'].max()),
        num=100)
    axes.hist(df_gamma_test['disp_y'], bins=bins, lw=3, histtype='step',
              label='Test $\gamma$')
    axes.hist(df_gamma_diffuse_test['disp_y'], bins=bins, lw=3, histtype='step',
              label='Test Diffuse $\gamma$')
    axes.hist(df_gamma_train['disp_y'], bins=bins, lw=3, histtype='step',
              label='Train $\gamma$')
    axes.hist(df_gamma_diffuse_train['disp_y'], bins=bins, lw=3,
              histtype='step', label='Train Diffuse $\gamma$')

    axes.set_xlabel('$DISP_y$ [mm]')
    axes.legend(loc='best')
    pdf.savefig(fig)

    fig = plt.figure()
    axes = fig.add_subplot(111)
    bins = np.linspace(
        min(df_gamma['disp_r'].min(), df_gamma_diffuse['disp_r'].min()),
        max(df_gamma['disp_r'].max(), df_gamma_diffuse['disp_r'].max()),
        num=100)
    axes.hist(df_gamma_test['disp_r'], bins=bins, lw=3, histtype='step',
              label='Test $\gamma$')
    axes.hist(df_gamma_diffuse_test['disp_r'], bins=bins, lw=3, histtype='step',
              label='Test Diffuse $\gamma$')
    axes.hist(df_gamma_train['disp_r'], bins=bins, lw=3, histtype='step',
              label='Train $\gamma$')
    axes.hist(df_gamma_diffuse_train['disp_r'], bins=bins, lw=3,
              histtype='step', label='Train Diffuse $\gamma$')
    axes.set_xlabel('$DISP_r$ [mm]')
    axes.legend(loc='best')
    pdf.savefig(fig)

    fig = plt.figure()
    axes = fig.add_subplot(111)
    bins = np.linspace(
        min(df_gamma['disp_theta'].min(), df_gamma_diffuse['disp_theta'].min()),
        max(df_gamma['disp_theta'].max(), df_gamma_diffuse['disp_theta'].max()),
        num=100)
    axes.hist(df_gamma_test['disp_theta'], bins=bins, lw=3, histtype='step',
              label='Test $\gamma$')
    axes.hist(df_gamma_diffuse_test['disp_theta'], bins=bins, lw=3, histtype='step',
              label='Test Diffuse $\gamma$')
    axes.hist(df_gamma_train['disp_theta'], bins=bins, lw=3, histtype='step',
              label='Train $\gamma$')
    axes.hist(df_gamma_diffuse_train['disp_theta'], bins=bins, lw=3,
              histtype='step', label='Train Diffuse $\gamma$')
    axes.set_xlabel(r'$DISP_{\theta}$ [rad]')
    axes.legend(loc='best')
    pdf.savefig(fig)

    axes = plot_feature_importance(features, importance)
    pdf.savefig(axes.get_figure())

    # dx = reco_direction_gamma_diffuse_test[:, 0] - df_gamma_diffuse_test_y[target[0]]
    dx = reco_direction_gamma_test[:, 0] - df_gamma_test_y[target[0]]
    # dy = reco_direction_gamma_diffuse_test[:, 1] - df_gamma_diffuse_test_y[target[1]]
    dy = reco_direction_gamma_test[:, 1] - df_gamma_test_y[target[1]]
    mean_x = np.mean(dx)
    mean_y = np.mean(dy)
    dr = np.sqrt((dx - mean_x)**2 + (dy - mean_y)**2)
    r_68 = np.quantile(dr, 0.68)

    x_bins = np.linspace(dx.min(), dx.max(), num=n_x)
    y_bins = np.linspace(dy.min(), dy.max(), num=n_y)

    label = '$R_{68} =$ '+'{:.2f} [mm]\n'.format(r_68) + \
            '$(x, y) =$ ({:.4f}, {:.4f}) [mm]'.format(mean_x, mean_y)
    axes = plot_hist_2d(dx, dy, bins=[x_bins, y_bins],
                        norm=LogNorm(), line=False,
                        label=label)
    axes.set_xlabel(r'$\Delta DISP_{x}$ [mm]')
    axes.set_ylabel(r'$\Delta DISP_{y}$ [mm]')
    # axes.legend(loc='best')
    axes.set_xlim((-500, 500))
    axes.set_ylim((-500, 500))
    axes.set_title('$Test \gamma$')
    print(label)
    pdf.savefig(axes.get_figure())

    x_intensity = np.linspace(np.log10(df_gamma_test_X['intensity']).min(), np.log10(df_gamma_test_X['intensity']).max(), num=20)
    r_68 = np.zeros(len(x_intensity) - 1) * np.nan

    for i in range(len(x_intensity) - 1):

        mask = (np.log10(df_gamma_test_X['intensity']) > x_intensity[i]) & (np.log10(df_gamma_test_X['intensity']) <= x_intensity[i+1])
        if mask.sum() < 100:
            continue

        r_68[i] = np.quantile(dr[mask], 0.68)

    r_68 = r_68 / FOCAL
    fig = plt.figure()
    axes = fig.add_subplot(111)
    x_intensity = x_intensity[:-1] + np.diff(x_intensity) * 0.5
    axes.plot(x_intensity, np.degrees(r_68))
    axes.set_xlabel(r'True $\log_{10}\left(\frac{size}{{\rm p.e.}}\right)$')
    axes.set_ylabel('$R_{68}$ [deg]')
    pdf.savefig(fig)

    x_energy = np.linspace(np.log10(df_gamma_test['true_energy']).min(),
                              np.log10(df_gamma_test['true_energy']).max(), num=20)
    x_centers = np.diff(x_energy) * 0.5 + x_energy[:-1]
    r_68 = np.zeros(len(x_energy) - 1) * np.nan
    r_68_jakub = np.zeros(len(x_energy) - 1) * np.nan
    median_x = np.zeros(len(x_energy) - 1) * np.nan
    median_y = np.zeros(len(x_energy) - 1) * np.nan
    mean_x = np.zeros(len(x_energy) - 1) * np.nan
    mean_y = np.zeros(len(x_energy) - 1) * np.nan


    for i in range(len(x_energy)-1):
        mask = (np.log10(df_gamma_test['true_energy']) > x_energy[i]) * (np.log10(df_gamma_test['true_energy']) <= x_energy[i+1])

        if mask.sum() < 100:
            continue

        r_68[i] = np.quantile(dr[mask], 0.68)
        median_x[i] = np.median(dx[mask])
        mean_x[i] = np.mean(dx[mask])
        median_y[i] = np.median(dy[mask])
        mean_y[i] = np.mean(dy[mask])

    r_68 = r_68 / FOCAL
    fig = plt.figure()
    axes = fig.add_subplot(111)
    axes.plot(x_centers, np.degrees(r_68), label='$R_{68}$')
    # axes.plot(x_centers, np.degrees(mean_x), label='Mean $x$')
    # axes.plot(x_centers, np.degrees(mean_y), label='Mean $y$')
    # axes.plot(x_centers, np.degrees(median_x), label='Median $x$')
    # axes.plot(x_centers, np.degrees(median_y), label='Median $y$')
    axes.set_xlabel(r'True $\log_{10}\left(\frac{E}{{\rm TeV}}\right)$')
    axes.set_ylabel('[deg]')
    axes.legend(loc='best')
    # axes.set_ylim((-0.3, 0.3))
    pdf.savefig(fig)

    dx = reco_direction_gamma_train[:, 0] - df_gamma_train_y[target[0]]
    dy = reco_direction_gamma_train[:, 1] - df_gamma_train_y[target[1]]
    mean_x = np.mean(dx)
    mean_y = np.mean(dy)
    dr = np.sqrt((dx - mean_x) ** 2 + (dy - mean_y) ** 2)
    r_68 = np.quantile(dr, 0.68)

    x_bins = np.linspace(dx.min(), dx.max(), num=n_x)
    y_bins = np.linspace(dy.min(), dy.max(), num=n_y)

    label = '$R_{68} =$ ' + '{:.2f} [mm]\n'.format(r_68) + \
            '$(x, y) =$ ({:.4f}, {:.4f}) [mm]'.format(mean_x, mean_y)
    axes = plot_hist_2d(dx, dy, bins=[x_bins, y_bins],
                        norm=LogNorm(), line=False,
                        label=label)
    axes.set_xlabel(r'$\Delta DISP_{x}$ [mm]')
    axes.set_ylabel(r'$\Delta DISP_{y}$ [mm]')
    # axes.legend(loc='best')
    axes.set_xlim((-500, 500))
    axes.set_ylim((-500, 500))
    axes.set_title('Train $\gamma$')
    print(label)
    pdf.savefig(axes.get_figure())

    dx = df_gamma_test['source_x'] - (reco_direction_gamma_test[:, 0] + df_gamma_test['x'])
    dy = df_gamma_test['source_y'] - (reco_direction_gamma_test[:, 1] + df_gamma_test['y'])
    mean_x = np.mean(dx)
    mean_y = np.mean(dy)
    dr = np.sqrt((dx - mean_x) ** 2 + (dy - mean_y) ** 2)
    r_68 = np.quantile(dr, 0.68)

    x_bins = np.linspace(dx.min(), dx.max(), num=n_x)
    y_bins = np.linspace(dy.min(), dy.max(), num=n_y)

    label = '$R_{68} =$ ' + '{:.2f} [mm]\n'.format(r_68) + \
            '$(x, y) =$ ({:.4f}, {:.4f}) [mm]'.format(mean_x, mean_y)
    axes = plot_hist_2d(dx, dy, bins=[x_bins, y_bins],
                        norm=LogNorm(), line=False,
                        label=label)
    axes.set_xlabel(r'$\Delta x_{source}$ [mm]')
    axes.set_ylabel(r'$\Delta x_{source}$ [mm]')
    # axes.legend(loc='best')
    axes.set_xlim((-500, 500))
    axes.set_ylim((-500, 500))
    axes.set_title('Test $\gamma$')
    print(label)
    pdf.savefig(axes.get_figure())

    plt.show()

output_filename = os.path.join(output_dir, 'direction_regressor.pk')

with open(output_filename, 'wb') as f:

    pass
    # pk.dump(rf_direction, f)
