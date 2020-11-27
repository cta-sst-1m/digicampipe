import argparse
from digicampipe.io.dl1 import read_hdf5, read_event_histogram
from matplotlib.backends.backend_pdf import PdfPages
import os
from digicampipe.visualization.machine_learning import plot_impact_parameter, \
    plot_effective_area, plot_3d_histo, LABELS, \
    plot_differential_trigger_rate, plot_impact_parameter_1d, \
    get_rate_gamma, get_rate_proton
import numpy as np
from digicampipe.pointing.disp import angular_distance


def make_3d_histogram(df, bins_energy, bins_impact, bins_theta):

    energy = np.array(np.log10(df['true_energy']))
    impact = np.array(np.sqrt(df['core_x']**2 + df['core_y']**2))
    theta = np.array(angular_distance(df['alt'], df['az'], df['tel_alt'], df['tel_az']))
    bins = [bins_energy, bins_impact, bins_theta]
    histo, _ = np.histogramdd([energy, impact, theta], bins=bins)
    return histo


def get_fitted(df):

    mask = np.isfinite(df['log_lh']) & np.isfinite(df['slope'])
    mask = mask & np.isfinite(df['intensity']) & np.isfinite(df['length'])
    mask = mask & np.isfinite(df['psi']) & np.isfinite(df['width'])
    mask = mask & np.isfinite(df['x']) & np.isfinite(df['y'])
    try:
        mask = mask & np.isfinite(df['intercept'])
    except KeyError:
        pass
    mask = np.arange(len(df))[mask]

    return df.iloc[mask]


if __name__ == '__main__':



    parser = argparse.ArgumentParser(description='Process some value.')
    parser.add_argument('--max_events', default=None, type=int,
                        help='sum the integers (default: find the max)')
    parser.add_argument('--size_cut', default=160,
                        type=float, help="A cut on size")
    parser.add_argument('--proton',  default='/sst1m/analyzed/mpeshowerfit/zenith_20_proton_v15.hdf5',
                        type=str, help='Processing version')
    parser.add_argument('--gamma', default='/sst1m/analyzed/mpeshowerfit/zenith_20_gamma_v15.hdf5',
                        type=str, help='Processing version')
    parser.add_argument('--gamma_diffuse',default='/sst1m/analyzed/mpeshowerfit/zenith_20_diffuse_gamma_v15.hdf5',
                        type=str, help='Processing version')
    parser.add_argument('--output_directory', default='.',
                        type=str, help='Processing version')


    args = parser.parse_args()
    max_events = args.max_events
    output_dir = args.output_directory

    gamma_file = args.gamma
    proton_file = args.proton
    gamma_diffuse_file = args.gamma_diffuse

    gamma_counts, bins_energy, bins_impact, bins_theta = read_event_histogram(gamma_file)
    bins_gamma = [bins_energy, bins_impact, bins_theta]

    gamma_rates = get_rate_gamma(bins_energy, bins_impact)
    df_gamma_triggered, factor = read_hdf5(gamma_file, max_events=max_events)
    gamma_counts = gamma_counts * factor
    weights_gamma = gamma_rates / gamma_counts.sum(axis=-1)
    mask = np.isfinite(weights_gamma)
    weights_gamma[~mask] = 0
    # weights_gamma /= np.sum(weights_gamma)

    gamma_counts_triggered = make_3d_histogram(df_gamma_triggered, bins_energy,
                                               bins_impact, bins_theta)


    df_gamma_fitted = get_fitted(df_gamma_triggered)
    gamma_counts_fitted = make_3d_histogram(df_gamma_fitted, bins_energy,
                                               bins_impact, bins_theta)

    print("Gamma sample loaded with {} simulated, {} triggered and {} fitted".format(np.sum(gamma_counts), len(df_gamma_triggered), len(df_gamma_fitted)))


    proton_counts, bins_energy, bins_impact, bins_theta = read_event_histogram(proton_file)
    bins_proton = [bins_energy, bins_impact, bins_theta]
    proton_rates = get_rate_proton(bins_energy, bins_impact, bins_theta)
    df_proton_triggered, factor = read_hdf5(proton_file, max_events=max_events)
    proton_counts = proton_counts * factor

    weights_proton = proton_rates / proton_counts
    mask = np.isfinite(weights_proton)
    weights_proton[~mask] = 0
    # weights_proton /= np.sum(weights_proton)

    proton_counts_triggered = make_3d_histogram(df_proton_triggered, bins_energy,
                                                bins_impact, bins_theta)
    df_proton_fitted = get_fitted(df_proton_triggered)
    proton_counts_fitted = make_3d_histogram(df_proton_fitted, bins_energy,
                                               bins_impact, bins_theta)
    print("Proton sample loaded with {} simulated, {} triggered and {} fitted".format(np.sum(proton_counts), len(df_proton_triggered), len(df_proton_fitted)))

    figure_path = os.path.join(output_dir, 'figure_effective_area.pdf')
    with PdfPages(figure_path) as pdf:

        axes = plot_3d_histo(gamma_rates, bins_gamma, label='Rate [Hz]')
        axes.set_xlabel(LABELS['log_true_energy'])
        axes.set_ylabel(LABELS['impact'])
        axes.set_title('Rate on-axis $\gamma$')
        pdf.savefig(axes.get_figure())

        axes = plot_3d_histo(gamma_counts, bins_gamma)
        axes.set_xlabel(LABELS['log_true_energy'])
        axes.set_ylabel(LABELS['impact'])
        axes.set_title('Simulated on-axis $\gamma$')
        pdf.savefig(axes.get_figure())

        axes = plot_3d_histo(weights_gamma, bins_gamma, label='Weights [Hz]')
        axes.set_xlabel(LABELS['log_true_energy'])
        axes.set_ylabel(LABELS['impact'])
        axes.set_title('On-axis $\gamma$')
        pdf.savefig(axes.get_figure())

        axes = plot_3d_histo(gamma_counts_triggered, bins_gamma)
        axes.set_xlabel(LABELS['log_true_energy'])
        axes.set_ylabel(LABELS['impact'])
        axes.set_title('Triggered on-axis $\gamma$')
        pdf.savefig(axes.get_figure())

        axes = plot_3d_histo(gamma_counts_fitted, bins_gamma)
        axes.set_xlabel(LABELS['log_true_energy'])
        axes.set_ylabel(LABELS['impact'])
        axes.set_title('Fitted on-axis $\gamma$')
        pdf.savefig(axes.get_figure())

        axes = plot_3d_histo(gamma_counts_triggered/gamma_counts, bins_gamma)
        axes.set_xlabel(LABELS['log_true_energy'])
        axes.set_ylabel(LABELS['impact'])
        axes.set_title('Trigger probability On-axis $\gamma$')
        pdf.savefig(axes.get_figure())

        axes = plot_3d_histo(gamma_counts_fitted/gamma_counts, bins_gamma)
        axes.set_xlabel(LABELS['log_true_energy'])
        axes.set_ylabel(LABELS['impact'])
        axes.set_title('Fit probability on-axis $\gamma$')
        pdf.savefig(axes.get_figure())

        axes = plot_impact_parameter_1d(gamma_counts, bins_impact=bins_impact)
        axes.set_title('Simulated on-axis $\gamma$')
        pdf.savefig(axes.get_figure())

        axes = plot_impact_parameter(df_gamma_triggered)
        axes.set_title('Triggered on-axis $\gamma$')
        pdf.savefig(axes.get_figure())

        axes = plot_impact_parameter(df_gamma_fitted)
        axes.set_title('Fitted on-axis $\gamma$')
        pdf.savefig(axes.get_figure())

        axes = plot_effective_area(gamma_counts,
                                   gamma_counts_triggered,
                                   bins_energy=bins_energy,
                                   bins_impact=bins_impact,
                                   label='Triggered', )

        axes = plot_effective_area(gamma_counts,
                                   gamma_counts_fitted,
                                   bins_energy=bins_energy,
                                   bins_impact=bins_impact,
                                   label='Fitted',
                                   axes=axes)
        axes.set_title('On-axis $\gamma$')
        axes.legend(loc='best')
        pdf.savefig(axes.get_figure())

        axes = plot_differential_trigger_rate(gamma_counts,
                                              gamma_counts_triggered,
                                              bins_energy=bins_energy,
                                              bins_impact=bins_impact,
                                              label='Crab triggered',
                                              kind='crab')

        axes = plot_differential_trigger_rate(gamma_counts,
                                              gamma_counts_fitted,
                                              bins_energy=bins_energy,
                                              bins_impact=bins_impact,
                                              label='Crab fitted', kind='crab',
                                              axes=axes)

        axes.set_title('On-axis $\gamma$')
        axes.legend(loc='best')
        pdf.savefig(axes.get_figure())

        axes = plot_3d_histo(proton_rates, bins_proton, label='Rate [Hz]')
        axes.set_xlabel(LABELS['log_true_energy'])
        axes.set_ylabel(LABELS['impact'])
        axes.set_title('Rate $p$')
        pdf.savefig(axes.get_figure())

        axes = plot_3d_histo(proton_rates, bins_proton, label='Rate [Hz]', axis=1)
        axes.set_xlabel(LABELS['log_true_energy'])
        axes.set_ylabel(LABELS['angular_distance'])
        axes.set_title('Rate $p$')
        pdf.savefig(axes.get_figure())

        axes = plot_3d_histo(proton_rates, bins_proton, label='Rate [Hz]',
                             axis=0)
        axes.set_xlabel(LABELS['impact'])
        axes.set_ylabel(LABELS['angular_distance'])
        axes.set_title('Rate $p$')
        pdf.savefig(axes.get_figure())

        axes = plot_3d_histo(proton_rates, bins_proton, proton_counts, label='Weights [Hz]')
        axes.set_xlabel(LABELS['log_true_energy'])
        axes.set_ylabel(LABELS['impact'])
        axes.set_title('Rate $p$')
        pdf.savefig(axes.get_figure())

        axes = plot_3d_histo(proton_rates, bins_proton, proton_counts, label='Weights [Hz]', axis=1)
        axes.set_xlabel(LABELS['log_true_energy'])
        axes.set_ylabel(LABELS['angular_distance'])
        axes.set_title('Rate $p$')
        pdf.savefig(axes.get_figure())

        axes = plot_3d_histo(proton_rates, bins_proton, proton_counts, label='Weights [Hz]',
                             axis=0)
        axes.set_xlabel(LABELS['impact'])
        axes.set_ylabel(LABELS['angular_distance'])
        axes.set_title('Rate $p$')
        pdf.savefig(axes.get_figure())

        axes = plot_3d_histo(proton_counts, bins_proton)
        axes.set_xlabel(LABELS['log_true_energy'])
        axes.set_ylabel(LABELS['impact'])
        axes.set_title('Simulated $p$')
        pdf.savefig(axes.get_figure())

        axes = plot_3d_histo(proton_counts, bins_proton, axis=1)
        axes.set_xlabel(LABELS['log_true_energy'])
        axes.set_ylabel(LABELS['angular_distance'])
        axes.set_title('Simulated $p$')
        pdf.savefig(axes.get_figure())

        axes = plot_3d_histo(proton_counts, bins_proton, axis=0)
        axes.set_xlabel(LABELS['impact'])
        axes.set_ylabel(LABELS['angular_distance'])
        axes.set_title('Simulated $p$')
        pdf.savefig(axes.get_figure())

        axes = plot_3d_histo(proton_counts_triggered, bins_proton)
        axes.set_xlabel(LABELS['log_true_energy'])
        axes.set_ylabel(LABELS['impact'])
        axes.set_title('Triggered $p$')
        pdf.savefig(axes.get_figure())

        axes = plot_3d_histo(proton_counts_triggered, bins_proton, axis=1)
        axes.set_xlabel(LABELS['log_true_energy'])
        axes.set_ylabel(LABELS['angular_distance'])
        axes.set_title('Triggered $p$')
        pdf.savefig(axes.get_figure())

        axes = plot_3d_histo(proton_counts_triggered, bins_proton, axis=0)
        axes.set_xlabel(LABELS['impact'])
        axes.set_ylabel(LABELS['angular_distance'])
        axes.set_title('Triggered $p$')
        pdf.savefig(axes.get_figure())

        axes = plot_3d_histo(proton_counts_fitted, bins_proton)
        axes.set_xlabel(LABELS['log_true_energy'])
        axes.set_ylabel(LABELS['impact'])
        axes.set_title('Fitted $p$')
        pdf.savefig(axes.get_figure())

        axes = plot_3d_histo(proton_counts_fitted, bins_proton, axis=1)
        axes.set_xlabel(LABELS['log_true_energy'])
        axes.set_ylabel(LABELS['angular_distance'])
        axes.set_title('Fitted $p$')
        pdf.savefig(axes.get_figure())

        axes = plot_3d_histo(proton_counts_fitted, bins_proton, axis=0)
        axes.set_xlabel(LABELS['impact'])
        axes.set_ylabel(LABELS['angular_distance'])
        axes.set_title('Fitted $p$')
        pdf.savefig(axes.get_figure())

        limit_energy = (3.15, np.inf)
        axes = plot_effective_area(proton_counts,
                                   proton_counts_triggered,
                                   bins_energy=bins_energy,
                                   bins_impact=bins_impact,
                                   bins_theta=bins_theta,
                                   limit_energy=limit_energy,
                                   label='Triggered', linestyle='-')

        axes = plot_effective_area(proton_counts,
                                   proton_counts_fitted,
                                   bins_energy=bins_energy,
                                   bins_impact=bins_impact,
                                   bins_theta=bins_theta,
                                   label='Fitted', linestyle='-',
                                   limit_energy=limit_energy,
                                   axes=axes)
        axes.set_title('Diffuse $p$')
        axes.legend(loc='best')
        pdf.savefig(axes.get_figure())

        axes = plot_3d_histo(proton_counts_triggered, counts_2=proton_counts, bins=bins_proton, axis=-1)
        axes.set_xlabel(LABELS['log_true_energy'])
        axes.set_ylabel(LABELS['impact'])
        axes.set_title('Trigger probability $p$')
        pdf.savefig(axes.get_figure())

        axes = plot_3d_histo(proton_counts_triggered, counts_2=proton_counts,
                             bins=bins_proton, axis=1)
        axes.set_xlabel(LABELS['log_true_energy'])
        axes.set_ylabel(LABELS['angular_distance'])
        axes.set_title('Trigger probability $p$')
        pdf.savefig(axes.get_figure())

        axes = plot_3d_histo(proton_counts_triggered, counts_2=proton_counts,
                             bins=bins_proton, axis=0)
        axes.set_xlabel(LABELS['impact'])
        axes.set_ylabel(LABELS['angular_distance'])
        axes.set_title('Trigger probability $p$')
        pdf.savefig(axes.get_figure())

        axes = plot_differential_trigger_rate(proton_counts,
                                              proton_counts_triggered,
                                              bins_energy=bins_energy,
                                              bins_impact=bins_impact,
                                              bins_theta=bins_theta,
                                              label='Proton triggered',
                                              kind='proton',
                                              limit_energy=limit_energy
                                              )

        axes = plot_differential_trigger_rate(proton_counts,
                                              proton_counts_fitted,
                                              bins_energy=bins_energy,
                                              bins_impact=bins_impact,
                                              bins_theta=bins_theta,
                                              label='Proton fitted',
                                              kind='proton',
                                              limit_energy=limit_energy,
                                              axes=axes)


        axes.legend(loc='best')
        pdf.savefig(axes.get_figure())

    print("Figure saved to {}".format(figure_path))