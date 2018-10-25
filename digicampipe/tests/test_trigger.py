import os
import numpy as np
from pkg_resources import resource_filename

from digicampipe.io.event_stream import event_stream
from digicampipe.calib.trigger import fill_trigger_input_7, fill_trigger_patch,\
    fill_digicam_baseline

example_file_path = resource_filename(
    'digicampipe',
    os.path.join(
        'tests',
        'resources',
        'SST1M_01_20180918_261.fits.fz'
    )
)


def test_trigger_input_7():
    events = event_stream([example_file_path])
    events = fill_trigger_input_7(events)
    for event in events:
        tel = event.r0.tels_with_data[0]
        assert np.all(np.isfinite(event.r0.tel[tel].trigger_input_7))


def test_compute_trigger_input_7():
    events = event_stream([example_file_path])
    events = fill_digicam_baseline(events)
    events = fill_trigger_patch(events)
    events = fill_trigger_input_7(events)
    for event in events:
        tel = event.r0.tels_with_data[0]
        assert np.all(np.isfinite(event.r0.tel[tel].trigger_input_7))


def test_compare_trigger_input_7():
    events_digi = event_stream([example_file_path], disable_bar=True)
    events_digi = fill_trigger_input_7(events_digi)

    events_comp = event_stream([example_file_path], disable_bar=True)
    events_comp = fill_digicam_baseline(events_comp)
    events_comp = fill_trigger_patch(events_comp)
    events_comp = fill_trigger_input_7(events_comp)
    for event_digi, event_comp in zip(events_digi, events_comp):
        tel = event_digi.r0.tels_with_data[0]
        ti7_digi = event_digi.r0.tel[tel].trigger_input_7
        ti7_comp = event_comp.r0.tel[tel].trigger_input_7
        abs_diff = np.abs(ti7_digi - ti7_comp)
        sum_ti7 = ti7_digi + ti7_comp
        assert np.mean(abs_diff) < 3
        assert np.nanmean(abs_diff/sum_ti7) < .05


if __name__ == '__main__':
    test_compare_trigger_input_7()