"""
Analysis the functionality of 108 optical modules and its total 1296 pixels
Usage:
  mts_analysis.py [options] [--] <INPUT>

Options:
  --help                        Show this
  <INPUT>                       PATH to the directory containing the list of "fits" files to analyse.
                                The first file (0000) is always the Dark Count.
                                The following are the different intensity levels starting by the lowest
  --module_id_list=LIST         List of the modules ID to be analyse.
  --output_dir=DIR              output directory for the module's PDFs
"""

import os
import sys
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
from ctapipe.visualization import CameraDisplay
from docopt import docopt
from digicampipe.utils.docopt import convert_list_int
from digicampipe.scripts.raw import compute
from astropy import units as u

from histogram.histogram import Histogram1D
from digicampipe.utils.pulse_template import NormalizedPulseTemplate


def mts_analysis(
        module_id_list, input_dir, output_dir
):

    if input_dir[-1] is not '/':
        input_dir = input_dir + '/'

    if output_dir[-1] is not '/':
        output_dir = output_dir + '/'

    try:
        os.mkdir(output_dir + module_id_list)
    except OSError:
        if os.path.isdir(output_dir + module_id_list) is True:
            print('Creation of the directory {} failed because it already exists'.format(output_dir + module_id_list))
        else:
            print('check for further problems')
    else:
        print('Successfully created the directory {}'.format(output_dir + module_id_list))


    [dir_path, dir_name, files] = os.walk(input_dir).__next__()
    files = [f for f in files if not f[0] == '.']
    files = sorted(files)
    print('{} of files in directory'.format(len(files)))

    ''' Creating names and label for future files and figures '''

    level_name = []
    for i, a_file in enumerate(files):
        if i == 0:
            level_name.append('dark_count')
        else:
            level_name.append('level_0{}'.format(i))

    output_fits_file_names = []
    for i, a_file in enumerate(files):
        output_fits_file = 'MOD_{}_{}_{}_{}_{}.fits'.format(module_id_list[0], module_id_list[1], module_id_list[2], module_id_list[3], level_name[i])
        print(output_dir + folders[0] + output_fits_file)
        output_fits_file_names.append(output_dir + folders[0] + output_fits_file)

    module_list = [372, 408, 409, 410, 444, 445, 446, 447, 480, 481, 482, 483, 484, 485,
                   516, 517, 518, 519, 520, 521, 552, 553, 554, 555, 556, 557, 588, 589,
                   590, 591, 592, 593, 624, 625, 626, 627, 628, 629, 661, 662, 663, 664,
                   665, 699, 700, 701, 736, 737]

    raw_histograms = []
    for i, a_file in enumerate(files):
        if i == 0:
            sample_range = [0, 25]
        else:
            sample_range = None

        histo = compute(files=dir_path + files[i],
                        filename=output_fits_file_names[i],
                        max_events=None,
                        pixel_id=module_list,
                        event_types=None,
                        disable_bar=False,
                        baseline_subtracted=False,
                        sample_range=sample_range)
        raw_histograms.append(histo)
        print('sample_range : ', sample_range)

    """
    raw_histograms = []
    for i, a_fits_file in enumerate(output_fits_file_names):
        raw_histograms.append(Histogram1D.load(output_fits_file_names[i]))
        #raw_histograms[0].draw()
    """


    pdf = PdfPages('/Users/lonewolf/Desktop/pdf_file.pdf')

    fig = plt.figure()
    axis = fig.add_subplot(111)
    for a_pixel in module_list:
        print(a_pixel)
        axis = raw_histograms[0].draw(index=a_pixel,
                                      errors=True,
                                      axis=axis,
                                      normed=False,
                                      log=True,
                                      legend=False,
                                      x_label='LSB',
                                      label='Raw dark in pixel {}'.format(a_pixel))
        fig.savefig(pdf, format='pdf')
        axis.clear()
    pdf.close()



    fadc_size = 48
    pixel_size = 12
    module_list_fadc = np.arange(1, fadc_size + 1, 1)
    pixel_list = np.arange(1, pixel_size + 1, 1)

    """ Creating the PDF for each module, which contains 12 pixels """

    for i, a_module in enumerate(module_id_list):
        print('Module {}'.format(a_module))
        pdf = PdfPages(output_dir + folders[1] + 'module_{}.pdf'.format(a_module))

        for j, a_pixel in enumerate(pixel_list):

            print('Pixel {}'.format(a_pixel))

        pdf.close()


def entry():
    args = docopt(__doc__)
    input_dir = args['<INPUT>']
    module_id_list = convert_list_int(args['--module_id_list'])
    output_dir = args['--output_dir']
    mts_analysis(
        input_dir, module_id_list, output_dir
    )


if __name__ == '__main__':
    entry()