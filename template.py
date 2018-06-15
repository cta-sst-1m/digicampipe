from os import path
import numpy as np
import pickle
import matplotlib.pyplot as plt
from scipy.interpolate import BSpline
import pkg_resources

templates_file_path = pkg_resources.resource_filename(
        'digicampipe',
        path.join(
            'tests',
            'resources',
            'templates_bspline.p'
        )
    )


def bspleval(x, knots, coeffs, order, debug=True):
    return BSpline(t=knots, c=coeffs, k=order, extrapolate=False, axis=0)(x)


def estimated_template(pe, start=0, stop=500, step=0.2):
    pkl_file = open(templates_file_path, 'rb')
    dict_template = pickle.load(pkl_file)
    xs = np.linspace(start, stop, (stop - start) * 1. / step)
    coeffs = []
    for coef in range(len(dict_template['coeff_sample'])):
        coeffs.append(dict_template['spline_coeff_func_pe'][coef](float(pe)))

    return xs, bspleval(
        x=xs,
        knots=dict_template['knots_sample'],
        coeffs=np.array(coeffs),
        order=5,
        debug=False
    )


def plot_pes_template(list_pe):
    plt.figure()
    for pe in list_pe:
        x_template, y_template = estimated_template(pe)
        plt.plot(
            x_template,
            y_template,
            '--',
            lw=2,
            label='$f(N_{\gamma}=%d)$' % pe)
    plt.legend()
    plt.show()


def amplitude():
    pes, gain, meas = [], [], []
    for logpe in np.arange(1., 4., 0.1):
        pe = 10.**logpe
        x_template, y_template = estimated_template(pe, start=0, stop=500)
        pes += [pe]
        gain += [np.nanmax(y_template) / pe / 4.72]
        meas += [np.nanmax(y_template) / 4.72]
    plt.figure()
    plt.semilogx(pes, gain)
    plt.show()
    plt.figure()
    plt.semilogx(pes, meas)
    plt.show()


def consecutive(data, stepsize=1):
    return np.split(data, np.where(np.diff(data) != stepsize)[0] + 1)


def integral():
    pes, gain, meas, integ, integ_2 = [], [], [], [], []
    for logpe in np.arange(1., 4., 0.1):
        pe = 10.**logpe
        x_template, y_template = estimated_template(pe, start=0, stop=500)
        pes += [pe]
        gain += [np.sum(y_template[:80]) / 5.]
        meas += [np.max(y_template) / 4.72]
        integ += [np.sum(y_template[:80]) / 5. / 22.]
        if meas[-1] > 570:
            good = consecutive(np.where(y_template > 0)[0])
            h = np.zeros((len(good), ))
            for i, g in enumerate(good):
                h[i] = np.sum(y_template[(g, )]) / 5. / 22.64
            integ_2 += [np.max(h)]
        else:
            integ_2 += [integ[-1]]
    print(pes, meas)
    plt.figure()
    plt.plot(pes, gain)
    plt.show()
    plt.figure()
    plt.plot(pes, integ, label='Integral')
    plt.plot(pes, integ_2, label='Integral until 0')
    plt.plot(pes, meas, label='Peak amplitude')
    plt.ylim(10., 10000.)
    plt.xlim(10., 10000.)
    plt.xlabel('$\mathrm{N_{true}(p.e.)}$')
    plt.ylabel('$\mathrm{N_{evaluated}(p.e.)}$')
    plt.legend()
    plt.show()


def plot_pe(pe):
    plt.figure()
    plt.xlabel('ADC')
    plt.ylabel('A.U.')
    plt.ylim(-400., 4096.)
    plt.xlim(0., 300.)
    x_template, y_template = estimated_template(pe, start=0, stop=500)
    #y_template[0:-11] = y_template[10:-1]
    plt.plot(
        x_template,
        y_template,
        'r',
        lw=2,
        label='$f(N_{\gamma}=%d),G=%0.3f$' % (pe, np.max(y_template) / pe))
    plt.legend()
    plt.show()


def dump_int_dat(pe):
    f = open('template_%s.dat' % str(pe), 'w')
    x_template, y_template = estimated_template(pe, start=0, stop=291, step=1)
    # Convert from PE to mV ; "Conversion factor 1PE = 2.4 mV"
    y_template = y_template * (-0.4285714285714286)
    f.write('-8.0 0.0\n')
    f.write('-7.0 0.0\n')
    f.write('-6.0 0.0\n')
    f.write('-5.0 0.0\n')
    f.write('-4.0 0.0\n')
    f.write('-3.0 0.0\n')
    f.write('-2.0 0.0\n')
    f.write('-1.0 0.0\n')
    f.write('0.0 0.0\n')
    for i in range(x_template.shape[0]):
        f.write('%0.1f %f\n' % (x_template[i], y_template[i]))
    f.close()


plot_pes_template([2000, 3000, 4000, 5000, 6000, 7000,
                   10000])  #,5,10,20,100,1000,4000])

dump_int_dat(3000)
dump_int_dat(4000)
dump_int_dat(5000)
dump_int_dat(6000)
dump_int_dat(7000)
dump_int_dat(8000)

amplitude()
#integral()
