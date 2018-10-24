import os
import numpy as np
from pkg_resources import resource_filename

from digicampipe.io.event_stream import event_stream
from digicampipe.calib.trigger import fill_trigger_input_7

example_file2_path = resource_filename(
    'digicampipe',
    os.path.join(
        'tests',
        'resources',
        'example_100_evts.000.fits.fz'
    )
)


def test_triger_input_7():
    events = event_stream([example_file2_path])
    events = fill_trigger_input_7(events)
    for event in events:
        tel = event.r0.tels_with_data[0]
        assert np.all(np.isfinite(event.r0.tel[tel].trigger_input_7))


if __name__ == '__main__':
    test_triger_input_7()
