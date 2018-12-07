import matplotlib.pyplot as plt
import numpy as np
from ctapipe.visualization import CameraDisplay
from scipy.stats import norm

from digicampipe.utils.pulse_template import NormalizedPulseTemplate
from digicampipe.instrument.camera import DigiCam


def plot_hillas(hillas_dict, title='', **kwargs):
    figure, axis_array = plt.subplots(3, 4)
    figure.suptitle(title)
    figure.subplots_adjust(top=0.95)
    plot_parameter(hillas_dict['cen_x'], name='$x$', units=' [mm]',
                   axis=axis_array[0, 0], **kwargs)
    plot_parameter(hillas_dict['cen_y'], name='$y$', units=' [mm]',
                   axis=axis_array[0, 1], **kwargs)
    plot_parameter(hillas_dict['length'], name='$l$', units=' [mm]',
                   axis=axis_array[0, 2], **kwargs)
    plot_parameter(hillas_dict['width'], name='$w$', units=' [mm]',
                   axis=axis_array[0, 3], **kwargs)
    plot_parameter(hillas_dict['phi'], name='$\phi$', units=' [rad]',
                   axis=axis_array[1, 0], **kwargs)
    plot_parameter(hillas_dict['psi'], name='$\psi$', units=' [rad]',
                   axis=axis_array[1, 1], **kwargs)
    plot_parameter(hillas_dict['miss'], name='miss', units=' [mm]',
                   axis=axis_array[1, 2], **kwargs)
    plot_parameter(hillas_dict['skewness'], name='skewness', units=' []',
                   axis=axis_array[1, 3], **kwargs, alpha=0.5)
    plot_parameter(hillas_dict['kurtosis'], name='kurtosis', units=' []',
                   axis=axis_array[1, 3], **kwargs, alpha=0.5)
    axis_array[1, 3].set_yscale('log')
    plot_parameter(hillas_dict['r'], name='r', units=' [mm]',
                   axis=axis_array[2, 0], **kwargs)
    plot_parameter(hillas_dict['alpha'], name=r'$\alpha$', units=' [rad]',
                   axis=axis_array[2, 1], **kwargs)
    plot_parameter(hillas_dict['width'] / hillas_dict['length'],
                   name=r'$\frac{w}{l}$', units=' [rad]',
                   axis=axis_array[2, 2], **kwargs)
    plot_parameter(hillas_dict['size'], name='size', units=' [p.e.]',
                   axis=axis_array[2, 3], log=True, **kwargs)

    return figure


def plot_parameter(parameter, name='', units='', axis=None, **kwargs):
    parameter = parameter[~(np.isnan(parameter)) * ~np.isinf(parameter)]

    if axis is None:

        plt.figure()
        plt.hist(parameter, label=name, **kwargs)
        plt.xlabel(name + units)
        plt.ylabel('count')
        plt.legend(loc='best')

    else:

        axis.hist(parameter, label=name + units, **kwargs)
        axis.legend(loc='best')


def plot_array_camera(data, label='', limits=None, **kwargs):
    mask = np.isfinite(data)

    if limits is not None:

        mask *= (data >= limits[0]) * (data <= limits[1])
    data = np.ma.masked_array(data, mask=~mask)

    fig = plt.figure()
    cam = DigiCam
    geom = cam.geometry
    cam_display = CameraDisplay(geom, **kwargs)
    cam_display.cmap.set_bad(color='k')
    cam_display.image = data
    cam_display.axes.set_title('')
    cam_display.axes.set_xticks([])
    cam_display.axes.set_yticks([])
    cam_display.axes.set_xlabel('')
    cam_display.axes.set_ylabel('')

    cam_display.axes.axis('off')
    cam_display.add_colorbar(label=label)
    cam_display.axes.get_figure().set_size_inches((10, 10))
    plt.axis('equal')
    if limits is not None:

        if not isinstance(limits, tuple):
            raise TypeError('Limits must be a tuple()')

        cam_display.colorbar.set_clim(vmin=limits[0], vmax=limits[1])

    cam_display.update()

    return cam_display, fig


def plot_correlation(x, y, c=None, label_x=' ', label_y=' ', label_c=' ',
                     **kwargs):
    mask = np.isfinite(x) * np.isfinite(y)

    x = x[mask]
    y = y[mask]

    if c is not None:
        c = c[mask]
    pearson_corr = np.corrcoef(x, y)[0, 1]

    plt.figure(figsize=(10, 10))
    plt.scatter(x, y, c=c, label='Correlation {:.02f}'.format(pearson_corr),
                **kwargs)

    if c is not None:
        plt.colorbar(label=label_c)
    plt.xlabel(label_x)
    plt.legend(loc='best')
    plt.ylabel(label_y)


def plot_histo(data, x_label='', show_fit=False, limits=None, **kwargs):
    mask = np.isfinite(data)

    if limits is not None:
        mask *= (data <= limits[1]) * (data >= limits[0])

    data = data[mask]
    mean = np.mean(data)
    std = np.std(data)

    try:

        data = data[~data.mask]

    except Exception:

        pass

    n_entries = len(data)

    label = '$N_{pixels}$ = ' + '{}\n'.format(n_entries)
    label_fit = 'Mean : {:.2f}\n'.format(mean) + 'Std : {:.2f}'.format(std)

    if not show_fit:
        label += label_fit

    fig = plt.figure(figsize=(10, 10))
    hist = plt.hist(data, **kwargs, label=label)

    if show_fit:
        bins = hist[1]

        bin_width = bins[1] - bins[0]
        x = np.linspace(np.min(data), np.max(data), num=len(data) * 10)
        y = n_entries * norm.pdf(x, loc=mean, scale=std) * bin_width
        plt.plot(x, y, color='r', label=label_fit)

    plt.xlabel(x_label)
    plt.ylabel('count')
    plt.legend(loc='best')

    return fig


def plot_pulse_templates(
        pulse_shape_files,
        axes=None, pixels=None, **kwargs
):
    if axes is None:
        fig, axes = plt.subplots(1, 1)
    else:
        fig = axes.get_figure()
    colors = plt.cm.rainbow(np.linspace(0, 1, len(pulse_shape_files)))
    for pulse_shape_file, color in zip(pulse_shape_files, colors):
        template = NormalizedPulseTemplate.create_from_datafiles(
            input_files=[pulse_shape_file],
            min_entries_ratio=0.1,
            pixels=pixels,
        )
        template.plot_interpolation(
            axes=axes,
            color=color,
            sigma=1,
            label=pulse_shape_file,
            **kwargs
        )
        del template
    axes.set_xlabel('time w.r.t half-height [ns]')
    axes.set_ylabel('normalized amplitude')
    return axes