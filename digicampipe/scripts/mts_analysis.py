#!/usr/bin/env python
"""
Do the parameters histogram for all the modules listed in a folder, each module contains 12 pixels and the idea is
to extract the fitted parameters (at maximum 1296 pixel) to display. The INPUT is the folder where all the individual modules are.

Usage:
    digicam-mts parameters --output=PATH --output_file=STR --ac_levels=INT [--debug --save_figure --fit_mpe_results=STR --fit_spe_results=STR --n_modules=INT --n_hw_pixel=INT] <INPUT>
    digicam-mts waveform_templates --output=PATH --ac_levels=INT --calib_file=FILE [--debug --save_figure --n_modules=INT --n_hw_pixel=INT --template_prefix=STR] <INPUT>
    digicam-mts charge_templates --output=PATH --ac_levels=INT --calib_file=FILE [--debug --save_figure --fit_mpe_results=STR --fit_spe_results=STR --n_modules=INT --n_hw_pixel=INT --template_prefix=STR] <INPUT>

Options:
    -h --help                   Show this screen.
    -v --debug                  Enter the debug mode.
    -o --output=PATH            Output folder.
                                [Default: ./mts_results]
    --output_file=STR           Calibration file name [Default: calibration_file.yml]
    --fit_mpe_results=STR       Name of the file with the fit results from multiple photon spectrum (digicampipe-mpe
                                fit combined).
                                [Default: fit_combine.fits]
    --fit_spe_results=STR       Name of the file with the fit from dark count spectra or spe (digicampipe-spe fit).
                                [Default: dc_spe_results.fits]
    --template_prefix=STR       Common prefix of the template.fits files on a module. Usually ac_levels + 1 number of template files, label from 0 to ac_levels. Level 0 is for the dark run. [Default: template_level].
    --n_modules=INT             Number of measured modules or input modules.
                                [Default: 108]
    --n_hw_pixel=INT            Number of pixel per measured or input module.
                                [Default: 12]
    --ac_levels=INT             number of LED AC DAC levels (this is not a list, unlike other scripts modules)
    --save_figure               Saves figures in pdf files in the ouptput folder. [Default: False]
    --calib_file=FILE           Provide the calibration file obtained from the parameters command. Unavoidable when using the commands waveform_templates or charge_templates.
                                [Defaut: None]
Commands:
    parameters                  Compute the histogram and make the yaml calibration file. Useful to get bad pixel in bad_pixel.py
    waveform_templates          Compute an average waveform's template from all pixels for all levels. Then make a chi_2_test distribution
    charge_templates            Compute an average charge's template from all pixels for all levels. Then make a chi_2_test distribution (TO BE COMPLETED)
"""

import os
import numpy as np
import yaml
import fitsio
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from docopt import docopt
from matplotlib.backends.backend_pdf import PdfPages
from digicampipe.visualization.plot import plot_histo, plot_array_camera
import digicampipe.utils.pulse_template as templates


def load_camera_config():

    # loading pixels id and coordinates from camera config file
    module_id, pixel_hw_id, pixel_sw_id = np.loadtxt(fname='digicampipe/digicampipe/tests/resources/camera_config.cfg',
                                                     unpack=True,
                                                     skiprows=47,
                                                     usecols=(2, 7, 8))

    module_id = module_id.astype(int)
    pixel_hw_id = pixel_hw_id.astype(int)
    pixel_sw_id = pixel_sw_id.astype(int)

    pixel_sw_id, module_id, pixel_hw_id = zip(*sorted(zip(pixel_sw_id, module_id, pixel_hw_id)))

    return pixel_sw_id, module_id, pixel_hw_id


def entry():
    args = docopt(__doc__)
    root = args['<INPUT>']
    output = args['--output']
    output_file = args['--output_file']
    debug = args['--debug']
    mpe_file = args['--fit_mpe_results']
    spe_file = args['--fit_spe_results']
    template_prefix = args['--template_prefix']
    n_modules = int(args['--n_modules'])
    n_hw_pixels = int(args['--n_hw_pixel'])
    n_light_levels = int(args['--ac_levels'])
    save_figure = args['--save_figure']
    calib_file = args['--calib_file']

    pixels_in_camera = 1296

    pixel_sw_id, module_id, pixel_hw_id = load_camera_config()
    config_table = np.zeros((pixels_in_camera, 3), dtype=int)
    config_table[:, 0] = module_id
    config_table[:, 1] = pixel_hw_id
    config_table[:, 2] = pixel_sw_id

    dir_list = [int(item) for item in os.listdir(root) if os.path.isdir(os.path.join(root, item))]
    dir_list = np.sort(np.array(dir_list))

    if not os.path.exists(output):
        os.makedirs(output)

    if args['parameters']:
        print('getting parameters from FITS files')

        # creation of empty dictionary
        calibration_data = {'baseline': np.zeros((pixels_in_camera,)),
                            'gain': np.zeros((pixels_in_camera,)),
                            'sigma_e': np.zeros((pixels_in_camera,)),
                            'sigma_s': np.zeros((pixels_in_camera,)),
                            'xt': np.zeros((pixels_in_camera,)),
                            'sw_pixel_id': np.zeros((pixels_in_camera,), dtype=int),
                            'dcr': np.zeros((pixels_in_camera,)),
                            'charge_chi2': np.zeros((pixels_in_camera,)),
                            'charge_ndf': np.zeros((pixels_in_camera,))}

        for level in range(n_light_levels):
            calibration_data['mean_{}'.format(level + 1)] = np.zeros((pixels_in_camera,))
            calibration_data['std_{}'.format(level + 1)] = np.zeros((pixels_in_camera,))
            calibration_data['pe_{}'.format(level + 1)] = np.zeros((pixels_in_camera,))

        for idx, pixel in enumerate(pixel_sw_id):

            mpe = '{}/{}/fits/{}'.format(root, module_id[idx], mpe_file)
            spe = '{}/{}/fits/{}'.format(root, module_id[idx], spe_file)

            with fitsio.FITS(mpe, 'r') as file:

                if debug:
                    print('index idx {} : sw-pix {}, mod {}, hw-pix {}'.format(idx, pixel_sw_id[idx], module_id[idx], pixel_hw_id[idx]))
                    print('file[MPE_COMBINED][key] index {} = hw-pix - 1'.format(pixel_hw_id[idx] - 1))
                    print(file['MPE_COMBINED'])

                # Pixel from 0 to 11 to be taken into account per module
                calibration_data['baseline'][pixel] = file['MPE_COMBINED']['baseline'].read()[pixel_hw_id[idx]-1]
                calibration_data['gain'][pixel] = file['MPE_COMBINED']['gain'].read()[pixel_hw_id[idx]-1]
                calibration_data['sigma_e'][pixel] = file['MPE_COMBINED']['sigma_e'].read()[pixel_hw_id[idx]-1]
                calibration_data['sigma_s'][pixel] = file['MPE_COMBINED']['sigma_s'].read()[pixel_hw_id[idx]-1]
                calibration_data['xt'][pixel] = file['MPE_COMBINED']['mu_xt'].read()[pixel_hw_id[idx]-1]
                calibration_data['charge_chi2'][pixel] = file['MPE_COMBINED']['chi_2'].read()[pixel_hw_id[idx]-1]
                calibration_data['charge_ndf'][pixel] = file['MPE_COMBINED']['ndf'].read()[pixel_hw_id[idx]-1]
                calibration_data['sw_pixel_id'][pixel] = pixel

            with fitsio.FITS(spe, 'r') as file:

                if debug:
                    print('index idx {} : sw-pix {}, mod {}, hw-pix {}'.format(idx, pixel_sw_id[idx], module_id[idx], pixel_hw_id[idx]))
                    print('file[SPE][key] index {} = hw-pix - 1'.format(pixel_hw_id[idx] - 1))
                    print(file['SPE'])

                # Pixel from 0 to 11 to be taken into account per module
                dcr_in_GHz = file['SPE']['dcr'].read()[pixel_hw_id[idx]-1]
                # this make the dcr in MHz
                calibration_data['dcr'][pixel] = dcr_in_GHz * 1e3

            with fitsio.FITS(mpe, 'r') as file:

                # An array of size "number of light intensity levels" per pixel
                for level in range(n_light_levels):
                    calibration_data['mean_{}'.format(level + 1)][pixel] = file['MPE_COMBINED']['mean'].read()[pixel_hw_id[idx] - 1][level]
                    calibration_data['std_{}'.format(level + 1)][pixel] = file['MPE_COMBINED']['std'].read()[pixel_hw_id[idx] - 1][level]
                    calibration_data['pe_{}'.format(level + 1)][pixel] = file['MPE_COMBINED']['mu'].read()[pixel_hw_id[idx] - 1][level]

        for key in calibration_data.keys():
            calibration_data[key] = calibration_data[key].tolist()

        yaml_file_path = '{}/{}'.format(output, output_file)
        with open(yaml_file_path, 'w') as outfile:
            yaml.dump(calibration_data, outfile, default_flow_style=True)

        print('calibration file {} was saved at {}'.format(output_file, output))

        if save_figure:
            labels = ['Baseline [LSB]', 'Gain [LSB / p.e.]', r'$\sigma_e$ [LSB]', r'$\sigma_s$ [LSB]',
                      r'$\mu_{XT}$ [p.e]', 'Software pixel ids', 'Dark Count Rate [MHz]', r'$\chi^2_{charge}$', r'$ndf_{charge}$']

            for level in range(n_light_levels):
                labels.append('Mean in Lvl. {} [LSB]'.format(level + 1))
                labels.append('STD in Lvl. {} [LSB]'.format(level + 1))
                labels.append('Number of p.e. in Lvl. {} [p.e.]'.format(level + 1))

            pdf_parameters = PdfPages('{}/parameters_on_camera.pdf'.format(output))

            temp_dict = {}
            for key, value in calibration_data.items():
                #if key not in ['charge_chi2', 'charge_ndf']:
                if key not in ['']:
                    temp_dict[key] = np.array(value)

            for k, key in enumerate(temp_dict.keys()):
                if debug:
                    print('idx : {}, label : {}, and key {}'.format(k, labels[k], key))

                histo_label = labels[k]
                mean = np.mean(temp_dict[key])
                std = np.std(temp_dict[key])
                fig = plot_histo(data=temp_dict[key], x_label=histo_label, bins=100)
                pdf_parameters.savefig(fig)
                plt.close(fig)
                print('histo {} saved'.format(labels[k]))

                cam_display, fig = plot_array_camera(data=temp_dict[key], label=histo_label)
                pdf_parameters.savefig(fig)
                plt.close(fig)
                print('cam display {} saved'.format(labels[k]))

            pdf_parameters.close()
            print('parameters_on_camera.pdf saved in {}'.format(output))

    if args['waveform_templates']:
        print('getting waveform templates from FITS files')

        if calib_file is not None:
            with open(calib_file) as file:
                calibration_parameters = yaml.load(file)
        else:
            print('calib_file.yml not found. Give a calibration file obtain from the parameters command')
            return

        for level in range(n_light_levels + 1):

            for idx, pixel in enumerate(pixel_sw_id):

                template_file = '{}/{}/fits/{}_{:02d}.fits'.format(root, module_id[idx], template_prefix, level)
                template = templates.NormalizedPulseTemplate.load(template_file)

                if debug:
                    with fitsio.FITS(template_file, 'r') as file:
                        print(file['PULSE_TEMPLATE'])

                if idx == 0:
                    template_list = []
                    template_list_err = []
                    template_mean = np.zeros(template.amplitude[pixel_hw_id[idx] - 1].shape)
                    template_std = np.zeros(template.amplitude_std[pixel_hw_id[idx] - 1].shape)

                # List of the 1296 templates
                template_list.append(template.amplitude[pixel_hw_id[idx] - 1])
                template_list_err.append(template.amplitude_std[pixel_hw_id[idx] - 1])
                # Making an average template recursively
                template_mean += template.amplitude[pixel_hw_id[idx] - 1]
                template_std += template.amplitude[pixel_hw_id[idx] - 1]**2

            template_mean /= (idx + 1.) # mean of x
            template_std /= (idx + 1.) # mean of x**2
            template_std = (idx + 1.)/(idx + 1. - 1.) * (template_std - template_mean**2)
            template_std = np.sqrt(template_std)

            # Making the object average template with NormalizedPulseTemplate
            averaged_template = templates.NormalizedPulseTemplate(template_mean, np.arange(50)*4, template_std)

            # Making chi_2 test for each level.
            test_array = []
            time_shift_array = []
            chi2_array = []
            # the templates might be shifted in time from the average template, so for best fit, we align them
            times = np.arange(50) * 4
            t_fit = np.linspace(-100, 100, num=1000)
            times = times - t_fit[..., None]
            y_template = averaged_template(times)

            for k, waveform in enumerate(template_list):

                chi2 = (waveform - y_template)**2 / template_list_err[k]**2
                chi2 = np.sum(chi2, axis=1)
                index_fit = np.argmin(chi2)

                test_array.append(chi2[index_fit])
                time_shift_array.append(t_fit[index_fit])
                chi2_array.append(chi2)

                if debug:
                    print('Waveform in pixel {}'.format(k))
                    print('fitted time shift : {} ns'.format(t_fit[index_fit]))
                    print('chi2 value in waveform : {}'.format(chi2[index_fit]))

            calibration_parameters['template_chi2_{}'.format(level)] = np.array(test_array).tolist()
            with open(calib_file, 'w') as file:
                yaml.dump(calibration_parameters, file, default_flow_style=True)

            if save_figure:
                pdf_waveforms = PdfPages('{}/{}_{}.pdf'.format(output, template_prefix, level))
                histo_label = r'$\chi^2$ test in level {}'.format(level)
                fig = plot_histo(data=np.array(test_array), x_label=histo_label, bins='auto')
                pdf_waveforms.savefig(fig)
                plt.close(fig)
                pdf_waveforms.close()

            if save_figure and debug:

                pdf_waveforms = PdfPages('{}/{}_{}.pdf'.format(output, template_prefix, level))
                t = np.linspace(0, 200 - 4, num=1000)
                t_waveform = np.arange(50) * 4
                y_array = []

                for time in time_shift_array:
                    y = averaged_template(t - time)
                    y_array.append(y)

                for j in range(0, len(template_list), 4):

                    fig, axs = plt.subplots(nrows=4, ncols=2, sharex='col')

                    ax = axs[0, 0]
                    ax.plot(t, y_array[j], color=sns.xkcd_rgb['red'],
                            label='Level {}, average'.format(level))
                    ax.plot(t_waveform, waveform, color=sns.xkcd_rgb['amber'],
                            label='Level {}, pixel {}'.format(level, j))
                    ax.set_ylabel('N. A.')
                    ax.legend(frameon=False, fontsize='x-small', loc=0)

                    ax = axs[0, 1]
                    ax.plot(t_fit, chi2_array[j], label=r'$\chi^2$')
                    ax.set_ylabel(r'$\chi^2$')

                    ax = axs[1, 0]
                    ax.plot(t, y_array[j+1], color=sns.xkcd_rgb['red'],
                            label='Level {}, average'.format(level))
                    ax.plot(t_waveform, waveform, color=sns.xkcd_rgb['amber'],
                            label='Level {}, pixel {}'.format(level, j+1))
                    ax.set_ylabel('N. A.')
                    ax.legend(frameon=False, fontsize='x-small', loc=0)

                    ax = axs[1, 1]
                    ax.plot(t_fit, chi2_array[j+1], label=r'$\chi^2$')
                    ax.set_ylabel(r'$\chi^2$')

                    ax = axs[2, 0]
                    ax.plot(t, y_array[j+2], color=sns.xkcd_rgb['red'],
                            label='Level {}, average'.format(level))
                    ax.plot(t_waveform, waveform, color=sns.xkcd_rgb['amber'],
                            label='Level {}, pixel {}'.format(level, j+2))
                    ax.set_ylabel('N. A.')
                    ax.legend(frameon=False, fontsize='x-small', loc=0)

                    ax = axs[2, 1]
                    ax.plot(t_fit, chi2_array[j+2], label=r'$\chi^2$')
                    ax.set_ylabel(r'$\chi^2$')

                    ax = axs[3, 0]
                    ax.plot(t, y_array[j+3], color=sns.xkcd_rgb['red'],
                            label='Level {}, average'.format(level))
                    ax.plot(t_waveform, waveform, color=sns.xkcd_rgb['amber'],
                            label='Level {}, pixel {}'.format(level, j+3))
                    ax.set_ylabel('N. A.')
                    ax.set_xlabel('time [ns]')
                    ax.legend(frameon=False, fontsize='x-small', loc=0)

                    ax = axs[3, 1]
                    ax.plot(t_fit, chi2_array[j+3], label=r'$\chi^2$')
                    ax.set_ylabel(r'$\chi^2$')
                    ax.set_xlabel('time [ns]')

                    pdf_waveforms.savefig(fig)
                    plt.close(fig)

                    if j == 100:
                        break

                fig, ax = plt.subplots()

                for k, waveform in enumerate(template_list):

                    ax.plot(t_waveform, waveform, color=sns.xkcd_rgb['amber'],
                            label='Waveform in pixel {}, in level {}'.format(k, level))
                    if waveform is template_list[-1]:
                        averaged = ax.plot(t, averaged_template(t), color=sns.xkcd_rgb['red'],
                                           label='Ave. waveform in level {}, over {} pixels'.format(level, idx + 1))
                        ax.legend(averaged, ('Ave. waveform in level {}, over {} pixels'.format(level, idx + 1),))
                        ax.set_xlabel('time [ns]')
                        ax.set_ylabel('Normalized Amplitude')
                    plt.close(fig)

                pdf_waveforms.savefig(fig)

                histo_label = r'$\chi^2$ test distribution'
                fig = plot_histo(data=np.array(test_array), x_label=histo_label, bins='auto')
                pdf_waveforms.savefig(fig)
                plt.close(fig)

                pdf_waveforms.close()

    if args['charge_templates']:
        # not implemented yet
        print('getting waveform templates from FITS files')

        if calib_file is not None:
            with open(calib_file) as file:
                calibration_parameters = yaml.load(file)
        else:
            print('calib_file.yml not found. Give a calibration file obtain from the parameters command')
            return

        for idx, pixel in enumerate(pixel_sw_id):

            template_file = '{}/{}/fits/{}_{:02d}.fits'.format(root, module_id[idx], template_prefix, level)
            template = templates.NormalizedPulseTemplate.load(template_file)

            if debug:
                with fitsio.FITS(template_file, 'r') as file:
                    print(file['PULSE_TEMPLATE'])

    return


if __name__ == '__main__':
    entry()
