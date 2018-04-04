import spe
import numpy as np
import pytest
from digicampipe.utils.exception import PeakNotFound
import numpy as np
import pytest

import spe
from digicampipe.utils.exception import PeakNotFound


def test_compute_gaussian_parameters_highest_peak():

    bins = np.arange(20)
    count = np.zeros(20)
    snr = 3
    count[10] = snr**2

    with pytest.raises(PeakNotFound):

        spe.compute_gaussian_parameters_highest_peak(bins, count, snr)

    spe.compute_gaussian_parameters_highest_peak(bins, count, snr - 0.1)


@pytest.mark.xfail
def test_build_raw_data_histogram():
    # test: spe.build_raw_data_histogram(events)
    assert False


@pytest.mark.xfail
def test_fill_histogram():
    # test: spe.fill_histogram(events, id, histogram)
    assert False


@pytest.mark.xfail
def test_find_pulse_1():
    # test: spe.find_pulse_1(events, threshold, min_distance)
    assert False


@pytest.mark.xfail
def test_find_pulse_2():
    # test spe.find_pulse_2(events, threshold_sigma, widths, **kwargs)
    assert False


@pytest.mark.xfail
def test_compute_charge():
    # test: spe.compute_charge(events, integral_width)
    assert False


@pytest.mark.xfail
def test_compute_amplitude():
    # test: spe.compute_amplitude(events)
    assert False


@pytest.mark.xfail
def test_spe_fit_function():
    # test: spe.spe_fit_function(x, baseline, gain, sigma_e, sigma_s, a_1, a_2, a_3, a_4)
    assert False


@pytest.mark.xfail
def test_compute_fit_init_param():
    # test: spe.compute_fit_init_param(x, y, snr=4, sigma_e=None, debug=False)
    assert False


@pytest.mark.xfail
def test_fit_spe():
    # test spe.fit_spe(x, y, y_err, snr=4, debug=False)
    assert False


@pytest.mark.xfail
def test_build_spe():
    # test spe.build_spe(events, max_events)
    assert False
