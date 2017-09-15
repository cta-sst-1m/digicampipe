import numpy as np
from scipy.interpolate import splrep, splev

nsb_rate = np.array([ 0.001     ,  0.00126896,
                      0.00161026,  0.00204336,
                      0.00259294,        0.00329034,
                      0.00417532,  0.00529832,
                      0.00672336,  0.00853168,
                      0.01082637,  0.01373824,
                      0.01743329,  0.02212216,
                      0.02807216,        0.03562248,
                      0.04520354,  0.05736153,
                      0.07278954,  0.09236709,
                      0.11721023,  0.14873521,
                      0.18873918,  0.23950266,
                      0.30391954,        0.38566204,
                      0.48939009,  0.62101694,
                      0.78804628,  1.        ])
std = np.array([  1.05399192,   1.10839055,
                   1.17901006,   1.24035378,
                   1.32706961,   1.43654254,
                   1.54922052,   1.69540818,
                   1.84739443,   2.0346974 ,
                   2.23064066,   2.48816963,
                   2.74982983,   3.06236889,
                   3.41572463,   3.78076031,
                   4.21579936,   4.67323316,
                   5.20949068,   5.75443471,
                   6.37877584,   6.98716587,
                   7.58629561,   8.30350721,
                   8.89673328,   9.50052861,
                   10.10113547,  10.55350981,
                   10.90199585,  11.1293928 ])
baseline = np.array([ 500.09329 ,  500.122574,
                      500.156548,  500.197976,
                      500.251504,        500.317864,
                      500.404168,  500.517638,
                      500.649846,  500.81971 ,
                      501.029466,  501.323184,
                      501.655576,  502.100004,
                      502.647232,        503.347018,
                      504.21708 ,  505.289202,
                      506.666586,  508.316356,
                      510.359676,  512.808806,
                      515.758656,  519.394132,
                      523.43345 ,        528.264008,
                      533.57916 ,  539.452982,
                      545.888096,  552.484512])

def compute_gain_drop(pedestal, type='std'):

    if type == 'std':
        spline = splrep(std, gain_drop(nsb_rate))

    elif type == 'mean':
        spline = splrep(baseline_shift, gain_drop(nsb_rate))
    else:
        raise('Unknown type %s' % type)
    return splev(pedestal, spline)

    return splev(pedestal, spline)


def compute_nsb_rate(pedestal, type='std'):

    if type == 'mean':

        spline = splrep(baseline_shift, nsb_rate)

    elif type == 'std':

        spline = splrep(std, nsb_rate)

    else:

        raise('Unknown type %s' % type)

    return splev(pedestal, spline)


def get_gains():

    return np.ones(1296) * 23. # TODO, replace gain of 23 by calib array of gain


def gain_drop(nsb_rate, cell_capacitance=85. * 1E-15, bias_resistance=10. * 1E3):

    return 1. / (1. + nsb_rate * cell_capacitance * bias_resistance * 1E9)


if __name__ == '__main__':

    import matplotlib.pyplot as plt

    data = np.load('true_mean_std_nsb.npz')

    nsb_rate = data['nsb_rate']
    nsb = data['nsb_rate']
    std = data['std']
    baseline_shift = data['mean'] - data['mean'][0]

    std_1 = np.linspace(0, 10, num=30)
    baseline_shift_1 = np.linspace(0, 50, num=30)

    plt.figure()
    plt.plot(std_1, compute_gain_drop(std_1, type='std'))
    plt.plot(std, gain_drop(nsb_rate), linestyle='None', marker='o')
    plt.legend()

    plt.figure()
    plt.plot(baseline_shift_1, compute_gain_drop(baseline_shift_1, type='mean'))
    plt.plot(baseline_shift, gain_drop(nsb_rate), linestyle='None', marker='o')
    plt.legend()

    plt.figure()
    plt.plot(std_1, compute_nsb_rate(std_1, type='std'))
    plt.plot(std, nsb_rate, linestyle='None', marker='o')
    plt.legend()

    plt.figure()
    plt.plot(baseline_shift_1, compute_nsb_rate(baseline_shift_1, type='mean'))
    plt.plot(baseline_shift, nsb_rate, linestyle='None', marker='o')
    plt.legend()

    plt.show()