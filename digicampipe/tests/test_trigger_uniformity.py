import os
import numpy as np
from pkg_resources import resource_filename
import tempfile

from digicampipe.scripts.trigger_uniformity import trigger_uniformity
from digicampipe.io.containers import CameraEventType


dark200_file_path = resource_filename(
    'digicampipe',
    os.path.join(
        'tests',
        'resources',
        'dark_200_evts.fits.fz'
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


def test_trigger_uniformity():
    # get triggers for science
    pixels_rate = trigger_uniformity(
        [science200_file_path], plot=None, event_types=None, disable_bar=True)
    assert not np.all(pixels_rate == 0)
    # get no triggers for dark
    pixels_rate = trigger_uniformity(
        [dark200_file_path], plot=None, event_types=None, disable_bar=True)
    assert np.all(pixels_rate == 0)
    # get no triggers for clock trigger events
    pixels_rate = trigger_uniformity(
        [science200_file_path],
        plot=None,
        event_types=CameraEventType.INTERNAL,
        disable_bar=True)
    assert np.all(pixels_rate == 0)


def test_trigger_uniformity_plot():
    files = [science200_file_path]
    with tempfile.TemporaryDirectory() as tmpdirname:
        plot_file=os.path.join(tmpdirname, 'test.png')
        pixels_rate = trigger_uniformity(files, plot=plot_file,
                                         event_types=None, disable_bar=True)
        assert os.path.isfile(plot_file)


if __name__ == '__main__':
    test_trigger_uniformity()
