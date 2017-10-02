import numpy as np
from scipy.interpolate import splrep, splev

baseline_shift = np.array([ 500.0876,   500.1667,   500.32636,  500.62612,  501.15552,  502.2427,
  503.95156,  507.1863,   512.9475,   521.82026,  535.0633,   551.6195,
  569.34392,  584.71116,  596.2085 ])

baseline_shift = baseline_shift - baseline_shift[0]

nsb_rate = np.array([  1.00000000e-03,   1.86822238e-03,   3.49025488e-03,   6.52057229e-03,
   1.21818791e-02,   2.27584593e-02,   4.25178630e-02,   7.94328235e-02,
   1.48398179e-01,   2.77240800e-01,   5.17947468e-01,   9.67641054e-01,
   1.80776868e+00,   3.37731391e+00,   6.30957344e+00])

std = np.array([  1.04232732,   1.22009471,   1.43720881,   1.80491932,   2.37300938,
   3.14624804,   4.12955852,   5.41114335,   7.05042862,   8.61511425,
  10.2820977,   11.06578871,  11.17625156,  10.75076423,  10.23973573]
)

"""
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
                   10.90199585,  11.1293928])

baseline_shift = np.array([ 00.09329 ,  00.122574,
                            00.156548,  00.197976,
                         0.251504,        00.317864,
                      00.404168,  00.517638,
                      00.649846,  00.81971 ,
                      01.029466,  01.323184,
                      01.655576,  02.100004,
                      02.647232,        03.347018,
                      04.21708 ,  05.289202,
                      06.666586,  08.316356,
                      10.359676,  12.808806,
                      15.758656,  19.394132,
                      23.43345 ,        28.264008,
                      33.57916 ,  39.452982,
                      45.888096,  52.484512])
"""
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