from digicampipe.scripts import spe


def test_compute_gaussian_parameters_highest_peak():
    # test: spe.compute_gaussian_parameters_highest_peak(
    #    bins, count, snr=4, debug=False)
    assert False


def test_build_raw_data_histogram():
    # test: spe.build_raw_data_histogram(events)
    assert False


def test_fill_histogram():
    # test: spe.fill_histogram(events, id, histogram)
    assert False


def test_find_pulse_1():
    # test: spe.find_pulse_1(events, threshold, min_distance)
    assert False


def test_find_pulse_2():
    # test spe.find_pulse_2(events, threshold_sigma, widths, **kwargs)
    assert False


def test_compute_charge():
    # test: spe.compute_charge(events, integral_width)
    assert False


def test_compute_amplitude():
    # test: spe.compute_amplitude(events)
    assert False


def test_spe_fit_function():
    # test: spe.spe_fit_function(x, baseline, gain, sigma_e, sigma_s, a_1, a_2, a_3, a_4)
    assert False


def test_compute_fit_init_param():
    # test: spe.compute_fit_init_param(x, y, snr=4, sigma_e=None, debug=False)
    assert False


def test_fit_spe():
    # test spe.fit_spe(x, y, y_err, snr=4, debug=False)
    assert False


def test_build_spe():
    # test spe.build_spe(events, max_events)
    assert False
