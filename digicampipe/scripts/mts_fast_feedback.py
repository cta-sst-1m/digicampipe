"""

MTS (Module Test Setup) test the camera modules.
The maximum modules the MTS can test each time is 4, which implies 48 pixels, for a pre-selected number of light
intensity N.
Number of files is N-1, since file_number runs from 0 to N-1, i.e. file 0 means Level 0 or in other words, Dark count.
However, here we do not run this raw data files

"""


import sys
import os
from histogram.histogram import Histogram1D
from digicampipe.utils.pulse_template import NormalizedPulseTemplate
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.offsetbox import AnchoredText
from matplotlib.backends.backend_pdf import PdfPages


def slash_it(directory):
    if directory[-1] is not '/':
        directory = directory + '/'
    return directory

if __name__ == '__main__':


    module_name = sys.argv[1]
    histo_dark = Histogram1D.load(sys.argv[2])
    histo_timing = Histogram1D.load(sys.argv[3])
    histo_mpe = Histogram1D.load(sys.argv[4])

    template_dark = NormalizedPulseTemplate.load(sys.argv[5])
    template_level_01 = NormalizedPulseTemplate.load(sys.argv[6])
    template_level_02 = NormalizedPulseTemplate.load(sys.argv[7])
    template_level_03 = NormalizedPulseTemplate.load(sys.argv[8])
    template_level_04 = NormalizedPulseTemplate.load(sys.argv[9])
    template_level_05 = NormalizedPulseTemplate.load(sys.argv[10])
    template_level_06 = NormalizedPulseTemplate.load(sys.argv[11])

    output_dir = sys.argv[12]

    # Naming of the files
    module_number = module_name.split('_')
    module_number = module_number[1:]

    for i, a_number in enumerate(module_number):
        module_number[i] = int(module_number[i])
    module_number = np.array(module_number)

    fig = plt.figure()
    axis = fig.add_subplot(111)

    output_dir = slash_it(output_dir) + module_name


    try:
        os.mkdir(output_dir)
    except OSError:
        if os.path.isdir(output_dir) is True:
            print('Creation of the directory {} failed because it already exists'.format(output_dir))
        else:
            print('check for further problems')
    else:
        print('Successfully created the directory {}'.format(output_dir))


    # mapping
    pixel_in_module_A = np.array([737, 701, 665, 629, 736, 700, 664, 628, 699, 663, 627, 591])
    pixel_in_module_B = np.array([593, 557, 521, 485, 592, 556, 520, 484, 555, 519, 483, 447])
    pixel_in_module_C = np.array([662, 626, 590, 554, 661, 625, 589, 553, 624, 588, 552, 516])
    pixel_in_module_D = np.array([518, 482, 446, 410, 517, 481, 445, 409, 480, 444, 408, 372])

    module_list = [pixel_in_module_A, pixel_in_module_B, pixel_in_module_C, pixel_in_module_D]

    for i, a_module in enumerate(module_list):
        print('This is module {}'.format(module_number[i]))

        for j, a_pixel in enumerate(module_list[i]):

            hardware_pixel_number = j+1
            software_pixel_number = a_pixel

            pdf = PdfPages('{}/MOD_{}_pixel_{}_(SW_pixel_{}).pdf'.format(output_dir,
                                                                         module_number[i],
                                                                         hardware_pixel_number,
                                                                         software_pixel_number))

            #print('DARK shape {}:'.format(histo_dark.shape))
            axis = histo_dark.draw(index=a_pixel,
                                   errors=True,
                                   axis=axis,
                                   normed=False,
                                   log=True,
                                   legend=False,
                                   x_label='LSB',
                                   label='Dark Count in MOD {}, pixel {}'.format(module_number[i], hardware_pixel_number))
            text = histo_dark._write_info(a_pixel)
            anchored_text = AnchoredText(text, loc=7)
            axis.add_artist(anchored_text)
            axis.legend(loc=1)
            fig.savefig(pdf, format='pdf')
            axis.clear()

            for k, a_intensity in enumerate(range(1, 7)):
                #print('TIMING shape {}'.format(histo_timing.shape))
                axis = histo_timing.draw(index=(k, a_pixel),
                                         errors=True,
                                         axis=axis,
                                         normed=False,
                                         log=False,
                                         legend=False,
                                         x_label='time',
                                         label='Peak time distribution for Level {}, pixel {}'.format(a_intensity, hardware_pixel_number))
                text = histo_timing._write_info((k, a_pixel))
                anchored_text = AnchoredText(text, loc=7)
                axis.add_artist(anchored_text)
                axis.legend(loc=1)
                fig.savefig(pdf, format='pdf')
                axis.clear()

                #print(histo_mpe.shape)
                axis = histo_mpe.draw(index=(k, a_pixel),
                                      errors=True,
                                      axis=axis,
                                      normed=False,
                                      log=False,
                                      legend=False,
                                      x_label='LSB',
                                      label='LBS count for Level {}, pixel {}'.format(a_intensity, hardware_pixel_number))
                text = histo_mpe._write_info((k, a_pixel))
                anchored_text = AnchoredText(text, loc=7)
                axis.add_artist(anchored_text)
                axis.legend(loc=1)
                fig.savefig(pdf, format='pdf')
                axis.clear()

            axis = template_dark[a_pixel].plot(axes=axis,
                                               label='Normalized template : Level 0 in MOD {}, pixel {}'.format(
                                                   module_number[i], hardware_pixel_number))
            fig.savefig(pdf, format='pdf')
            axis.clear()

            axis = template_level_01[a_pixel].plot(axes=axis,
                                               label='Normalized template : Level 1 in MOD {}, pixel {}'.format(
                                                   module_number[i], hardware_pixel_number))
            fig.savefig(pdf, format='pdf')
            axis.clear()

            axis = template_level_02[a_pixel].plot(axes=axis,
                                               label='Normalized template : Level 2 in MOD {}, pixel {}'.format(
                                                   module_number[i], hardware_pixel_number))
            fig.savefig(pdf, format='pdf')
            axis.clear()

            axis = template_level_03[a_pixel].plot(axes=axis,
                                               label='Normalized template : Level 3 in MOD {}, pixel {}'.format(
                                                   module_number[i], hardware_pixel_number))
            fig.savefig(pdf, format='pdf')
            axis.clear()

            axis = template_level_04[a_pixel].plot(axes=axis,
                                               label='Normalized template : Level 4 in MOD {}, pixel {}'.format(
                                                   module_number[i], hardware_pixel_number))
            fig.savefig(pdf, format='pdf')
            axis.clear()

            axis = template_level_05[a_pixel].plot(axes=axis,
                                               label='Normalized template : Level 5 in MOD {}, pixel {}'.format(
                                                   module_number[i], hardware_pixel_number))
            fig.savefig(pdf, format='pdf')
            axis.clear()

            axis = template_level_06[a_pixel].plot(axes=axis,
                                               label='Normalized template : Level 6 in MOD {}, pixel {}'.format(
                                                   module_number[i], hardware_pixel_number))
            fig.savefig(pdf, format='pdf')
            axis.clear()


            #or k, a_intensity in enumerate(range(1, 7)):
            #    axis = template_level_06[a_pixel].plot(axes=axis,
            #                                           label='Normalized template : Level 6 in MOD {}, pixel {}'.format(module_number[i], hardware_pixel_number))
            #    fig.savefig(pdf, format='pdf')
            #    axis.clear()

            pdf.close()


