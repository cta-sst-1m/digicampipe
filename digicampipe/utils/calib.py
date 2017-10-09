import numpy as np
from scipy.interpolate import splrep, splev

baseline_shift = np.array([ 500.10844,  500.13456,  500.19772,  500.2697,   500.44418,  500.6094,
                                             500.87226,  501.33124,  501.8768,   502.71168,  503.9574,   505.51968,
                                             507.92038,  511.24914,  515.572,    521.41952,  528.5894,   537.03064,
                                             547.04614,  557.76222,  568.03408,  577.88066,  586.55614,  593.50884,
                                             598.97944,  603.33174,  606.54438,  608.84342,  610.4056,   611.94482])

baseline_shift = baseline_shift - baseline_shift[0]

nsb_rate = np.array([1.00000000e-03,   1.45222346e-03,   2.10895298e-03,   3.06267099e-03,
                                      4.44768267e-03,   6.45902911e-03,   9.37995361e-03,   1.36217887e-02,
                                      1.97818811e-02,   2.87277118e-02,   4.17190571e-02,   6.05853935e-02,
                                      8.79835297e-02,   1.27771746e-01,   1.85553127e-01,   2.69464604e-01,
                                      3.91322820e-01,   5.68288180e-01,   8.25281427e-01,   1.19849305e+00,
                                      1.74047972e+00,   2.52756549e+00,   3.67058990e+00,   5.33051677e+00,
                                      7.74110151e+00,   1.12418092e+01,   1.63256191e+01,   2.37084470e+01,
                                      3.44299630e+01,   5.00000000e+01])

std =np.array([1.10364884,   1.12430139,   1.2435541,    1.36286533,   1.59703604,
                                 1.80362736,   2.07318656,   2.50829027,   2.92629147,   3.44292195,
                                 4.0800227,    4.79139778,   5.63669767,   6.53108791,   7.64749998,
                                 8.65923108,   9.57379588,  10.37193237,  10.88006117,  11.19791948,
                                11.10610816,  10.95768853,  10.57739894,  10.37384412,  10.18064523,
                                10.75009621,  11.57963516,  13.24342791,  15.57271873,  18.24045984])


def gain_drop(nsb_rate, cell_capacitance=85. * 1E-15, bias_resistance=10. * 1E3):

    return 1. / (1. + nsb_rate * cell_capacitance * bias_resistance * 1E9)


# spline_std_gain_drop = splrep(std, gain_drop(nsb_rate))
spline_mean_gain_drop = splrep(baseline_shift, gain_drop(nsb_rate))
# spline_std_nsb_rate = splrep(std, nsb_rate)
spline_mean_nsb_rate = splrep(baseline_shift, nsb_rate)


def compute_gain_drop(pedestal, type='mean'):

    if type == 'mean':

        return splev(pedestal, spline_mean_gain_drop)
    else:
        raise('Unknown type %s' % type)


def compute_nsb_rate(pedestal, type='mean'):

    if type == 'mean':

        return splev(pedestal, spline_mean_nsb_rate)

    else:

        raise('Unknown type %s' % type)


def get_gains():

    return np.ones(1296) * 23. # TODO, replace gain of 23 by calib array of gain


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