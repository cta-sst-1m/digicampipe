import os
import tempfile

import numpy as np
from pkg_resources import resource_filename

from digicampipe.scripts.get_burst import entry as get_burst


example_file2_path = resource_filename(
    'digicampipe',
    os.path.join(
        'tests',
        'resources',
        'example_100_evts.000.fits.fz'
    )
)


def test_get_burst():
    files = [example_file2_path]
    with tempfile.TemporaryDirectory() as tmpdirname:
        output1 = os.path.join(tmpdirname, "test1.txt")
        output2 = os.path.join(tmpdirname, "test2.txt")

        # test with high threshold
        no_burst = False
        try:
            get_burst(
                files, plot_baseline="none", event_average=100, threshold_lsb=2.,
                output=output1, expand=10, merge_sec=5., video_prefix="none"
            )
        except SystemExit:
            no_burst = True
        assert no_burst

        # test with low threshold
        no_burst = False
        try:
            get_burst(
                files, plot_baseline="none", event_average=100,
                threshold_lsb= 1e-4,
                output=output2, expand=10, merge_sec=5., video_prefix="none"
            )
        except SystemExit:
            no_burst = True
        assert no_burst is False

        assert os.path.isfile(output1) is False  # no file created if no burst
        assert os.path.isfile(output2)

if __name__ == '__main__':
    test_get_burst()
