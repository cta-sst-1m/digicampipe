import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib as mpl
from matplotlib.colors import LogNorm
import numpy as np
from histogram.histogram import Histogram1D
from skimage.util.shape import view_as_windows as viewW
from sklearn.preprocessing import normalize


def strided_indexing_roll(a, r):
    # Concatenate with sliced to cover all rolls
    a_ext = np.concatenate((a, a[:, :-1]), axis=1)
    # Get sliding windows; use advanced-indexing to select appropriate ones
    n = a.shape[1]
    return viewW(a_ext, (1, n))[np.arange(len(r)), (n-r) % n, 0]


def entry(files):

    raw_adc = Histogram1D.load(files[0])
    sum_raw_adc = np.zeros(shape=raw_adc.shape)
    fig0 = plt.figure()
    color_m = cm.get_cmap('viridis')

    all_std = np.zeros(shape=raw_adc.std().shape)
    list_std = []

    for i, f in enumerate(files):
        #Recover all histos from raw ADC waveform
        raw_adc = Histogram1D.load(f)
        mean = raw_adc.mean()

        #plt.subplot(len(files), 1, i+1)
        date = f[24:34].replace('_', ' ') + ', mean = %.02f, $\sigma$ = %.02f' % (np.mean(mean), np.std(mean))
        col_curve = color_m(float(i)/float(len(files)))
        plt.hist(mean, bins=range(240, 340), color=col_curve, fill=False, histtype='step', linewidth=2, label=date)

        std = raw_adc.std()
        a = np.subtract(np.ones(shape=(1296, ))*200, mean).astype(dtype=int)
        sum_raw_adc += strided_indexing_roll(raw_adc.data, a)
        list_std.extend(std)
        all_std += std

    plt.legend()
    plt.xlim(240, 340)
    plt.xlabel('mean baseline [LSB]')
    plt.ylabel('#channels/LSB')
    plt.yscale('log')
    plt.show()

    sum_dark = np.zeros(shape=(4094,))
    normed_matrix = normalize(sum_raw_adc, axis=1, norm='max')

    fig1 = plt.figure()
    cnt = 0
    list_pix = []
    #Sum off all pixels
    for i in range(1296):
        good = False
        if all_std[i] < 21:
            if sum_raw_adc[i][190] <1e4:
                if sum_raw_adc[i][193] < 5e5:
                    if sum_raw_adc[i][194] < 1e4:
                        if cnt==0:
                            plt.plot(np.arange(4094), normed_matrix[i], linewidth=0.5, color='black', alpha=0.2,
                                     label='All pixels, sum all runs')
                        else:
                            plt.plot(np.arange(4094), normed_matrix[i], linewidth=0.5, color='black', alpha=0.2)
                        sum_dark += sum_raw_adc[i]
                        cnt += 1
                        good = True
        if not good:
            list_pix.append(i)


    normed_matrix = np.delete(normed_matrix, list_pix, axis=0)
    mean_matrix = np.average(normed_matrix, axis=0)

    plt.plot(np.arange(4094), mean_matrix, color='r', label = 'Average')
    plt.ylim(1e-6, 2e0)
    plt.xlim(180, 260)
    plt.yscale('log')
    plt.xlabel('raw waveform [LSB]')
    #plt.ylabel('normalized counts')
    plt.ylabel('normalized counts')
    plt.legend()

    color_m = cm.get_cmap('viridis')
    color_list = []
    for i in range(1296):
       color_list.append(color_m(float(i)/float(1296)))

    fig2 = plt.figure(figsize=(14, 12))
    ax = fig2.add_subplot(111)
    #plt.hist(list_std, bins=1000)
    #plt.yscale('log')
    #plt.xlim(0, 5)
    #plt.xlabel('std raw LSB projection [LSB]')
    #plt.ylabel('counts')
    #plt.text(3, 1000, 'mean = %.3f' % np.mean(std), fontsize=22)
    #plt.text(3, 500, '$\sigma$ = %.3f' % np.std(std), fontsize=22)
    #plt.imshow([[0] * 1296, range(1296)], cmap='viridis', interpolation='none')
    #plt.gca().set_visible(False)
    #for i in range(len(cum_sum)):
        #plt.plot(np.arange(4094), cum_sum[len(cum_sum)-i], color=color_m(float(i)/float(len(cumsum))))
    #    plt.fill_between(np.arange(4094), np.zeros(shape=(4094, )), cum_sum[len(cum_sum)-i-1],
    #                     color=color_m(float(i)/1296))
    #plt.colorbar(ticks=range(1296))

    col = ax.stackplot(np.arange(4094), np.delete(sum_raw_adc, list_pix, axis=0), colors=color_list)

    plt.ylim(1e1, 2e10)
    plt.xlim(180, 260)
    plt.xlabel('raw waveform [LSB]')
    plt.ylabel('cumulative counts')
    #fig2.colorbar(col)
    #plt.colorbar()

    plt.show()

    return


if __name__ == '__main__':

    files = [
        'data/dark_run/raw_histo_2018_08_16.pk',
        'data/dark_run/raw_histo_2018_08_17.pk',
        'data/dark_run/raw_histo_2018_08_18.pk',
        'data/dark_run/raw_histo_2018_08_19.pk',
        'data/dark_run/raw_histo_2018_08_20.pk',
        'data/dark_run/raw_histo_2018_08_21.pk',
        'data/dark_run/raw_histo_2018_08_22.pk',
        'data/dark_run/raw_histo_2018_08_23.pk',
        'data/dark_run/raw_histo_2018_08_29.pk',
        'data/dark_run/raw_histo_2018_09_01.pk',
        'data/dark_run/raw_histo_2018_09_02.pk',
        'data/dark_run/raw_histo_2018_09_19.pk'
    ]

    #files = ['data/dark_run/raw_histo_2018_08_16.pk']
    # 'data/dark_run/raw_histo_2018_08_29.pk',

    entry(files)