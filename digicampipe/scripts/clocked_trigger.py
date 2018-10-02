import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib as mpl
from matplotlib.colors import LogNorm
import numpy as np
from histogram.histogram import Histogram1D
from skimage.util.shape import view_as_windows as viewW
from sklearn.preprocessing import normalize
from digicampipe.utils import DigiCam
from digicampipe.utils.geometry import compute_patch_matrix
from ctapipe.visualization import CameraDisplay
import matplotlib.animation as animation


def strided_indexing_roll(a, r):
    # Concatenate with sliced to cover all rolls
    a_ext = np.concatenate((a, a[:, :-1]), axis=1)
    # Get sliding windows; use advanced-indexing to select appropriate ones
    n = a.shape[1]
    return viewW(a_ext, (1, n))[np.arange(len(r)), (n-r) % n, 0]


def animate_baseline_shift(baseline_shifts, video, zmin, zmax):

    fig = plt.figure()
    ax = plt.gca()
    display = CameraDisplay(DigiCam.geometry, ax=ax)
    display.set_limits_minmax(zmin, zmax)
    display.add_colorbar()
    title = plt.title("")

    display.image = baseline_shifts[0]

    def update(i):
        display.image = baseline_shifts[i]
        return display, title

    nframe = len(baseline_shifts)
    print('creating animation with', nframe , 'frames')
    anim = animation.FuncAnimation(fig, update,
                                   frames=np.arange(1, nframe),
                                   interval=200)
    print('saving...')
    if video != "show":
        anim.save(video, writer='ffmpeg',
                  metadata=dict(artist='matthieu.heller@unige.ch',
                                comment='baseline shift variation from run to run'),
                  codec='mpeg4', bitrate=20000)
        print('video saved as', video)
    else:
        plt.show()
    plt.close(fig)

def entry(files, file_dark, file_dict):

    raw_adc_dark = Histogram1D.load(file_dark)
    mean_dark = raw_adc_dark.mean()

    print(mean_dark)

    data_dict = {}
    size = []
    az = []
    el = []

    mask = np.zeros(shape=(1296,), dtype=int)

    run_list = []

    for i, f in enumerate(files):
        print(f)
        #Recover all histos from raw ADC waveform
        raw_adc = Histogram1D.load(f)
        current_run = int(f[51:54])
        source = file_dict[f[51:54]]['Source']
        data_dict.setdefault(source, {'Runs': [], 'Mean': [], 'Std': [], 'Shift': [], 'Std_perPix': {},
                                      'Shift_perPix': {}})

        mean = raw_adc.mean()
        std = raw_adc.std()
        data_dict[source]['Runs'].append(current_run)
        data_dict[source]['Mean'].extend(mean)
        data_dict[source]['Std'].extend(raw_adc.std())
        shift = np.subtract(mean, mean_dark)

        data_dict[source]['Shift'].extend(shift)
        for j in range(1296):
            data_dict[source]['Shift_perPix'].setdefault(str(j), [])
            data_dict[source]['Shift_perPix'][str(j)].append(shift[j])
            data_dict[source]['Std_perPix'].setdefault(str(j), [])
            data_dict[source]['Std_perPix'][str(j)].append(std[j])

        mask += (shift < 0).astype(int)

        size.append(np.average(raw_adc.mean()))
        if float(file_dict[f[51:54]]['Az']) < 0:
            last_az = 360. + float(file_dict[f[51:54]]['Az'])
        else:
            last_az = float(file_dict[f[51:54]]['Az'])
        last_el = 90.-float(file_dict[f[51:54]]['El'])
        az.append(last_az)
        el.append(last_el)

        '''
        fig0 = plt.figure()
        #fig0 = plt.figure(figsize=(16, 8))
        #ax = plt.subplot(1, 2, 1)
        ax = plt.gca()
        display = CameraDisplay(DigiCam.geometry, ax=ax)
        display.add_colorbar()
        display.image = raw_adc.std()
        display.set_limits_minmax(6, 10)
        ax.set_title('Std of raw ADC distribution', fontsize=18)
        plt.text(200, 550, 'Source: %s' % file_dict[f[51:54]]['Source'], fontsize=24)
        plt.text(-500, 575, 'Start: %s' % file_dict[f[51:54]]['Start'], fontsize=24)
        plt.text(-500, 525, 'Stop: %s' % file_dict[f[51:54]]['Stop'], fontsize=24)

        plt.text(200, 480, 'Source: %s' % file_dict[f[51:54]]['Source'], fontsize=14)
        plt.text(-475, 510, 'Start: %s' % file_dict[f[51:54]]['Start'], fontsize=12)
        plt.text(-475, 470, 'Stop: %s' % file_dict[f[51:54]]['Stop'], fontsize=12)
        ax1 = plt.subplot(122, polar=True)
        plt.scatter(az, el, s=size)
        plt.scatter(last_az, last_el, color='red')
        ax1.set_rmax(90)
        ax1.set_theta_zero_location("N")
        ax1.set_theta_direction(-1)
        ax1.set_yticks(np.arange(0, 90 + 15, 15))
        ax1.set_yticklabels(ax1.get_yticks()[::-1])

        plt.savefig(f[:len(f)-3]+'_std.png')
        plt.close()

        fig0 = plt.figure()
        ax = plt.gca()
        display = CameraDisplay(DigiCam.geometry, ax=ax, title='Mean of raw ADC distribution')
        display.add_colorbar()
        display.image = raw_adc.mean()
        #plt.show()
        plt.text(200, 550, 'Source: %s' % file_dict[f[51:54]]['Source'], fontsize=24)
        plt.text(-500, 575, 'Start: %s' % file_dict[f[51:54]]['Start'], fontsize=24)
        plt.text(-500, 525, 'Stop: %s' % file_dict[f[51:54]]['Stop'], fontsize=24)
        display.set_limits_minmax(280, 390)
        plt.savefig(f[:len(f) - 3] + '_mean.png')
        plt.close()
        '''

    color_m = cm.get_cmap('viridis')

    fig1 = plt.figure()
    for i, k in enumerate(data_dict.keys()):
        plt.hist(data_dict[k]['Std'], bins=np.linspace(0, 20, 101), color=color_m(float(i)/float(len(data_dict))), linewidth=2, alpha=0.5,
                 label=k)
    plt.xlabel('mean per run [LSB]')
    plt.ylabel('#pixels')
    plt.yscale('log')
    plt.legend()
    plt.savefig('/Users/mheller/Documents/CTA/ctasoft/digicampipe/digicampipe/scripts/data/clocked_trigger/StdDist_perSource.png')
    plt.close()

    fig2 = plt.figure()
    for i, k in enumerate(data_dict.keys()):
        plt.hist(data_dict[k]['Mean'], bins=range(250, 401), color=color_m(float(i)/float(len(data_dict))), linewidth=2, alpha=0.5,
                 label=k)
    plt.xlabel('mean per run [LSB]')
    plt.ylabel('#pixels')
    plt.yscale('log')
    plt.legend()
    plt.savefig('/Users/mheller/Documents/CTA/ctasoft/digicampipe/digicampipe/scripts/data/clocked_trigger/MeanDist_perSource.png')
    plt.close()

    for i, m in enumerate(mask):
        if m > 0:
            print(i)

    fig3 = plt.figure()
    plt.plot(np.arange(0, 1296), mask, linestyle='', marker='o', markersize=2)
    plt.xlabel('Channel number')
    plt.ylabel('#Runs (total = 336)')

    figa = plt.figure()
    plt.scatter(data_dict['1ES']['Shift_perPix']['600'], data_dict['1ES']['Std_perPix']['600'],
                c=data_dict['1ES']['Runs'], cmap='viridis')
    plt.xlabel('Baseline shift [LSB]')
    plt.ylabel('Baseline std [LSB]')
    plt.colorbar()

    figa1 = plt.figure()
    for i in range(1296):
        plt.plot(data_dict['1ES']['Shift_perPix'][str(i)], data_dict['1ES']['Std_perPix'][str(i)], markersize=2,
                 linestyle='', marker='o', color=color_m(float(i)/float(1296)))
    plt.xlabel('Baseline shift [LSB]')
    plt.ylabel('Baseline std [LSB]')

    figc = plt.figure()
    for i in range(1296):
        plt.plot(data_dict['1ES']['Runs'], data_dict['1ES']['Shift_perPix'][str(i)], markersize=2, marker='o',
                 color=color_m(float(i)/float(1296)))
    plt.xlabel('File number')
    plt.ylabel('Baseline shift [LSB]')

    figc1 = plt.figure()
    for i in range(1296):
        plt.plot(data_dict['1ES']['Runs'][1:], np.ediff1d(data_dict['1ES']['Shift_perPix'][str(i)]), markersize=2, marker='o',
                 color=color_m(float(i) / float(1296)))
    plt.xlabel('File number')
    plt.ylabel('baseline shift variation [LSB]')

    '''
    figb = plt.figure()
    plt.scatter(data_dict['Crab']['Shift_perPix']['600'], data_dict['Crab']['Std_perPix']['600'],
                c=data_dict['Crab']['Runs'], cmap='viridis')
    plt.xlabel('Baseline shift [LSB]')
    plt.ylabel('Baseline std [LSB]')
    plt.colorbar()

    figb1 = plt.figure()
    for i in range(1296):
        plt.plot(data_dict['Crab']['Shift_perPix'][str(i)], data_dict['Crab']['Std_perPix'][str(i)], markersize=2,
                 linestyle='', marker='o', color=color_m(float(i)/float(1296)))
    plt.xlabel('Baseline shift [LSB]')
    plt.ylabel('Baseline std [LSB]')

    figd = plt.figure()
    for i in range(1296):
        plt.plot(data_dict['Crab']['Runs'], data_dict['Crab']['Shift_perPix'][str(i)], markersize=2, marker='o',
                 color=color_m(float(i) / float(1296)))
    plt.xlabel('File number')
    plt.ylabel('Baseline shift [LSB]')

    figd1 = plt.figure()
    for i in range(1296):
        plt.plot(data_dict['Crab']['Runs'][1:], np.ediff1d(data_dict['Crab']['Shift_perPix'][str(i)]), markersize=2,
                 marker='o',
                 color=color_m(float(i) / float(1296)))
    plt.xlabel('File number')
    plt.ylabel('baseline shift variation [LSB]')

    figb = plt.figure()
    plt.scatter(data_dict['Crab']['Shift_perPix']['600'], data_dict['Crab']['Std_perPix']['600'],
                c=data_dict['Crab']['Runs'], cmap='viridis')
    plt.xlabel('Baseline shift [LSB]')
    plt.ylabel('Baseline std [LSB]')
    plt.colorbar()

    figb1 = plt.figure()
    for i in range(1296):
        plt.plot(data_dict['Crab']['Shift_perPix'][str(i)], data_dict['Crab']['Std_perPix'][str(i)], markersize=2,
                 linestyle='', marker='o', color=color_m(float(i)/float(1296)))
    plt.xlabel('Baseline shift [LSB]')
    plt.ylabel('Baseline std [LSB]')
    '''

    # building array for animation
    baseline_shift_var = np.zeros(shape=(len(data_dict['1ES']['Runs'])-1, 1296), dtype=float)
    for i in range(len(data_dict['1ES']['Runs'][1:])):
        for j in range(1296):
            baseline_shift_var[i][j] = np.ediff1d(data_dict['1ES']['Shift_perPix'][str(j)])[i]

    animate_baseline_shift(baseline_shift_var, 'bl_shift_var_anim.mp4', zmin=-50, zmax=50)

    return

if __name__ == '__main__':

    files = []
    info_file = open('data/clocked_trigger/files_info.txt')
    all_lines = info_file.readlines()
    all_lines = [x.strip() for x in all_lines]
    file_dict = {}
    for l in all_lines:
        if l[0] != '#':
            l = l.split(' ')
            if l[4] != 'None':
                run_num = l[0][56:59]
                file_dict[run_num] = {'Run': None, 'Start': None, 'Stop': None, 'Source': None, 'Az': None, 'El': None}
                file_dict[run_num]['Run'] = run_num
                file_dict[run_num]['Start'] = l[2].replace('T', ' ')
                file_dict[run_num]['Stop'] = l[3].replace('T', ' ')
                file_dict[run_num]['Source'] = l[4]
                if l[4] == '1ES':
                    file_dict[run_num]['El'] = l[6].replace('(', '').replace(',', '')
                    file_dict[run_num]['Az'] = l[7].replace(')', '')
                else:
                    file_dict[run_num]['El'] = l[5].replace('(', '').replace(',', '')
                    file_dict[run_num]['Az'] = l[6].replace(')', '')

    for i in range(301, 602):
    #for i in range(301, 305):
        files.append('data/clocked_trigger/clocked_raw_SST1M_01_20180919_%03i.pk' % i)


    entry(files, 'data/dark_run/raw_histo_2018_09_19.pk', file_dict)
