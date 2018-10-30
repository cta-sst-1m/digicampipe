import os
import numpy as np
from pkg_resources import resource_filename

from digicampipe.scripts.trigger_uniformity import entry as trigger_uniformity


example_file2_path = resource_filename(
    'digicampipe',
    os.path.join(
        'tests',
        'resources',
        'example_100_evts.000.fits.fz'
    )
)


def test_data_quality():
    files = [example_file2_path]
    pixels_rate = trigger_uniformity(files, plot=None, event_types=None)
    # the events in the example files are dark event, so no trigger is expected
    assert np.all(pixels_rate == 0)


if __name__ == '__main__':
    test_data_quality()
