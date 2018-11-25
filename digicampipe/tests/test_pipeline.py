import os
import tempfile
import numpy as np
from astropy.io import fits
from pkg_resources import resource_filename

from digicampipe.scripts.pipeline import main_pipeline
from digicampipe.scripts.raw import compute as compute_raw
from digicampipe.utils.docopt import convert_pixel_args
from digicampipe.scripts.plot_pipeline import plot_pipeline, \
    get_data_and_selection, scan_2d_plot

dark100_file_path = resource_filename(
    'digicampipe',
    os.path.join(
        'tests',
        'resources',
        'dark_100_evts.fits.fz'
    )
)
dark200_file_path = resource_filename(
    'digicampipe',
    os.path.join(
        'tests',
        'resources',
        'dark_200_evts.fits.fz'
    )
)
science100_file_path = resource_filename(
    'digicampipe',
    os.path.join(
        'tests',
        'resources',
        'run150_100_evts.fits.fz'
    )
)
science200_file_path = resource_filename(
    'digicampipe',
    os.path.join(
        'tests',
        'resources',
        'run150_200_evts.fits.fz'
    )
)
calibration_filename = resource_filename(
    'digicampipe',
    os.path.join(
        'tests',
        'resources',
        'calibration_20180814.yml'
    )
)
template_filename = resource_filename(
    'digicampipe',
    os.path.join(
        'tests',
        'resources',
        'pulse_template_all_pixels.txt'
    )
)
aux_basepath = os.path.dirname(science100_file_path)

expected_columns = ['local_time', 'event_id', 'event_type', 'az', 'el',
                    'phi', 'y', 'skewness', 'intensity', 'x', 'psi', 'width',
                    'length', 'r', 'kurtosis','miss', 'alpha',
                    'baseline', 'nsb_rate', 'digicam_temperature',
                    'pdp_temperature', 'target_ra', 'target_dec',
                    'pointing_leds_on', 'pointing_leds_blink', 'all_hv_on',
                    'all_ghv_on', 'is_on_source', 'is_tracking',
                    'shower', 'border', 'burst', 'saturated']


def test_pipeline():
    # checks that the pipeline produce a fits file with all columns
    with tempfile.TemporaryDirectory() as tmpdirname:
        dark_filename = os.path.join(tmpdirname, 'dark.pk')
        hillas_filename = os.path.join(tmpdirname, 'hillas.fits')
        compute_raw(
            files=[dark100_file_path],
            max_events=None,
            pixel_id=convert_pixel_args(None),
            filename=dark_filename,
            disable_bar=True
        )
        main_pipeline(
            files=[science100_file_path],
            aux_basepath=aux_basepath,
            max_events=None,
            dark_filename=dark_filename,
            integral_width=7,
            debug=False,
            hillas_filename=hillas_filename,
            template_filename=template_filename,
            parameters_filename=calibration_filename,
            picture_threshold=30,
            boundary_threshold=15,
            saturation_threshold=3000,
            threshold_pulse=0.1,
            disable_bar=True
        )
        hdul = fits.open(os.path.join(tmpdirname, 'hillas.fits'))
        cols = [c.name for c in hdul[1].columns]
        nevent = len(hdul[1].data['local_time'])
        assert nevent > 0
        for col in expected_columns:
            assert col in cols, 'column "{}" was not found.'.format(col)
            assert len(hdul[1].data[col]) == nevent
        data = hdul[1].data
        good_data = np.isfinite(data.intensity)
        # all data where Hillas computation succeeded have some intensity.
        assert np.all(data.intensity[good_data] > 90)
        # no shower are close to the center in the test ressources.
        assert np.all((data.r[good_data] > 50) & (data.r[good_data] < 650))
        # cog of showers are within the camera
        assert np.all((data.x[good_data] < 500) & (data.x[good_data] > -500))
        assert np.all((data.y[good_data] < 550) & (data.y[good_data] > -550))


def test_pipeline_plot():
    with tempfile.TemporaryDirectory() as tmpdirname:
        dark_filename = os.path.join(tmpdirname, 'dark.pk')
        hillas_filename = os.path.join(tmpdirname, 'hillas.fits')
        compute_raw(
            files=[dark200_file_path],
            max_events=None,
            pixel_id=convert_pixel_args(None),
            filename=dark_filename,
            disable_bar=True
        )
        main_pipeline(
            files=[science200_file_path],
            aux_basepath=aux_basepath,
            max_events=None,
            dark_filename=dark_filename,
            integral_width=7,
            debug=False,
            hillas_filename=hillas_filename,
            template_filename=template_filename,
            parameters_filename=calibration_filename,
            picture_threshold=30,
            boundary_threshold=15,
            bad_pixels=None,
            saturation_threshold=3000,
            threshold_pulse=0.1,
            disable_bar=True
        )
        hdul = fits.open(os.path.join(tmpdirname, 'hillas.fits'))
        nevent = len(hdul[1].data['local_time'])
        assert nevent > 0
        shower_plot = os.path.join(tmpdirname, 'shower_center.png')
        hillas_plot = os.path.join(tmpdirname, 'hillas.png')
        cor_all_plot = os.path.join(tmpdirname, 'correlation_all.png')
        cor_sel_plot = os.path.join(tmpdirname, 'correlation_selected.png')
        cor_cut_plot = os.path.join(tmpdirname, 'correlation_cut.png')
        plot_pipeline(
            hillas_filename,
            cut_length_gte=None,
            cut_length_lte=25,
            cut_width_gte=None,
            cut_width_lte=15,
            cut_length_over_width_gte=10,
            cut_length_over_width_lte=2,
            cut_border_eq=None,
            cut_burst_eq=None,
            cut_saturated_eq=None,
            alphas_min=None,
            plot_scan2d=None,
            plot_showers_center=shower_plot,
            plot_hillas=hillas_plot,
            plot_correlation_all=cor_all_plot,
            plot_correlation_selected=cor_sel_plot,
            plot_correlation_cut=cor_cut_plot,
        )
        assert os.path.isfile(shower_plot)
        assert os.path.isfile(hillas_plot)
        assert os.path.isfile(cor_all_plot)
        assert os.path.isfile(cor_sel_plot)
        assert os.path.isfile(cor_cut_plot)

        # we plot scan_2d apart to have more control on the scan, so it can
        # be fast
        scan2d_plot = os.path.join(tmpdirname, 'scan2d.png')
        data, selection = get_data_and_selection(
            hillas_file=hillas_filename,
            cut_length_lte=25,
            cut_width_lte=15,
            cut_length_over_width_gte=10,
            cut_length_over_width_lte=2,
        )
        scan_2d_plot(data[selection], alphas_min=[5.], num_steps=3,
                     plot=scan2d_plot)
        assert os.path.isfile(scan2d_plot)


if __name__ == '__main__':
    test_pipeline()
    test_pipeline_plot()
