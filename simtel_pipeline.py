import numpy as np
import matplotlib.pyplot as plt
# import digicampipe.io.hessio_digicam as hsm     #
import astropy.units as u
from digicampipe.io.event_stream import event_stream
from digicampipe.utils import geometry
from cts_core.camera import Camera
from digicampipe.utils import utils
from digicampipe.io.save_hillas import save_hillas_parameters_in_text, \
    save_hillas_parameters

from digicampipe.calib.camera import dl0, dl2, filter, r1, dl1
from digicampipe.utils import utils, calib
import simtel_baseline      #
import events_image         #
import mc_shower            #

from optparse import OptionParser
import os


# import inspect
# print(inspect.getfile(dl2))

if __name__ == '__main__':

    parser = OptionParser()
    parser.add_option("-p", "--path", dest="directory",
                      help="directory to data files",
                      default='../../../sst-1m_simulace/data_test/ryzen_testprod/0.0deg/Data/')
    parser.add_option("-o", "--output", dest="output",
                      help="output filename", default="hillas", type=str)
    parser.add_option('-c', "--camera_config",
                      dest='camera_config_file',
                      help='camera config file to load Camera()',
                      default='digicampipe/tests/resources/camera_config.cfg')
    parser.add_option('-i', "--picture_threshold", dest='picture_threshold',
                      help='Picture threshold', default=15, type=int)
    parser.add_option('-b', "--boundary_threshold", dest='boundary_threshold',
                      help='Boundary threshold', default=7, type=int)
    parser.add_option('-z', "--zenit", dest='zenit_angle',
                      help='Zenit distance, THETAP',
                      default=0, type=int)
    parser.add_option('-a', "--azimut", dest='azimut', help='Azimut, PHIP',
                      default=0, type=int)
    parser.add_option('-r', "--primary", dest='primary',
                      help='Primary particle', default='gamma', type=str)
    parser.add_option('-d', "--baseline0", dest='baseline_beginning',
                      help='N bins from the beginning of the waveform for \
                      baseline calculation', type=int, default=9)
    parser.add_option('-e', "--baseline1", dest='baseline_end',
                      help='N bins from the end of the waveform for baseline \
                      calculation', type=int, default=15)

    (options, args) = parser.parse_args()

    digicam_config_file = 'camera_config_digicampipe.cfg'

    # Input/Output files
    directory = options.directory
    all_file_list = os.listdir(directory)
    file_list = []
    string1 = (options.primary + '_' + str(options.zenit_angle) +
               'deg_' + str(options.azimut) + 'deg_')
    print(string1)
    for fi in all_file_list:
        if (string1 in fi and '___cta-prod3-sst-dc-2150m--sst-dc' in fi and
                '.simtel.gz' in fi):
            print(fi)
            file_list.append(directory + fi)

    digicam_config_file = options.camera_config_file
    dark_baseline = None

    # Source coordinates (in camera frame)
    source_x = 0. * u.mm
    source_y = 0. * u.mm

    # Camera and Geometry objects
    # (mapping, pixel, patch + x,y coordinates pixels)
    digicam = Camera(_config_file=digicam_config_file)
    digicam_geometry = geometry.generate_geometry_from_camera(
                        camera=digicam,
                        source_x=source_x,
                        source_y=source_y)

    # Noisy pixels not taken into account in Hillas
    pixel_not_wanted = [
        1038, 1039, 1002, 1003, 1004, 966, 967, 968, 930, 931, 932, 896]
    additional_mask = np.ones(1296)
    additional_mask[pixel_not_wanted] = 0
    additional_mask = additional_mask > 0

    # Integration configuration (signal reco.)
    time_integration_options = {'mask': None,
                                'mask_edges': None,
                                'peak': None,
                                'window_start': 3,
                                'window_width': 15,  # length of integration window
                                'threshold_saturation': np.inf,
                                'n_samples': 50,
                                'timing_width': 10,
                                'central_sample': 11}

    peak_position = utils.fake_timing_hist(
        time_integration_options['n_samples'],
        time_integration_options['timing_width'],
        time_integration_options['central_sample'])

    (
        time_integration_options['peak'],
        time_integration_options['mask'],
        time_integration_options['mask_edges']
    ) = utils.generate_timing_mask(
        time_integration_options['window_start'],
        time_integration_options['window_width'],
        peak_position)

    # Image cleaning configuration
    picture_threshold = options.picture_threshold
    boundary_threshold = options.boundary_threshold
    shower_distance = 200 * u.mm

    # Filtering on big showers
    min_photon = 50

    # Config for Hillas parameters analysis
    n_showers = 100000000
    reclean = True

    # Define the event stream
    data_stream = event_stream(
        filelist=file_list,
        camera_geometry=digicam_geometry,
        camera=digicam)

    # create data_stream
    # data_stream = hsm.hessio_event_source(
    #    data,
    #    camera_geometry=digicam_geometry,
    #    camera=digicam)

    # Clean pixels
    data_stream = filter.set_pixels_to_zero(
        data_stream, unwanted_pixels=pixel_not_wanted)

    # Computation of baseline
    #
    # Methods:
    #
    # 'data'
    # Baseline is computed as a mean of 'n_bins0' first time samples,
    # 'n_bins1' last time samples.
    # A key assumption in this method is that the shower in simulated
    # data is somewhere in the middle of 50 samples.
    # Each pixel in each event has its own baseline
    #
    # 'simtel'
    # Baseline is taken as simulated value event.mc.pedestal/50 that can
    # be set up with the use of variable 'fadc_pedestal'
    # in CTA-ULTRA6-SST-DC.cfg.

    data_stream = simtel_baseline.fill_baseline_r0(
        data_stream, n_bins0=options.baseline_beginning,
        n_bins1=options.baseline_end, method='data')

    # Run the r1 calibration (i.e baseline substraction)
    data_stream = r1.calibrate_to_r1(data_stream, dark_baseline)

    # Run the dl1 calibration (compute charge in photons + cleaning)
    data_stream = dl1.calibrate_to_dl1(
        data_stream,
        time_integration_options,
        additional_mask=additional_mask,
        picture_threshold=picture_threshold,
        boundary_threshold=boundary_threshold)

    # Return only showers with total number of p.e. above min_photon
    data_stream = filter.filter_shower(data_stream, min_photon=min_photon)

    # Suffix for output filenames
    suffix = (
              options.primary +
              '_ze' + str(options.zenit_angle).zfill(2) +
              '_az' + str(options.azimut).zfill(3) +
              '_p' + str(options.picture_threshold).zfill(2) +
              '_b' + str(options.boundary_threshold).zfill(2)
             )

    # Save cleaned events - pixels and corresponding values
    filename_pix = 'pixels.txt'
    filename_eventsimage = 'events_image_' + suffix + '.txt'
    data_stream = events_image.save_events(
        data_stream, directory + filename_pix,
        directory + filename_eventsimage)

    # Save simulated shower paramters
    filename_showerparam = 'pipedmc_param_' + suffix + '.txt'
    data_stream = mc_shower.save_shower(
        data_stream, directory + filename_showerparam)

    # Run the dl2 calibration (Hillas)
    data_stream = dl2.calibrate_to_dl2(
        data_stream, reclean=reclean,
        shower_distance=shower_distance)

    # Save arrival times of photons in pixels passed cleaning
    filename_timing = 'timing_' + suffix + '.txt'
    data_stream = events_image.save_timing(
        data_stream, directory + filename_timing)

    # Save mean baseline in event pixels
    filename_baseline = (
                         'baseline_' + suffix +
                         '_bas' + str(options.baseline_beginning).zfill(2) +
                         str(options.baseline_end).zfill(2) + '.txt'
                         )
    # data_stream = simtel_baseline.save_mean_event_baseline(
    #    data_stream, directory + filename_baseline)

    # for event in data_stream:
    #    print(event.dl0.event_id)

    # Save Hillas
    hillas_filename = options.output + '_' + suffix
    save_hillas_parameters(
        data_stream=data_stream,
        n_showers=n_showers,
        output_filename=directory + hillas_filename)
    # save_hillas_parameters_in_text(
    #    data_stream=data_stream,
    #    output_filename=directory + hillas_filename)

    """
    import matplotlib.pyplot as plt
    for event in data_stream:
        telescope_id = 1
        dl1_camera = event.dl1.tel[telescope_id]
        geom = event.inst.geom[telescope_id]
        pix_x = np.asanyarray(geom.pix_x, dtype=np.float64).value
        pix_y = np.asanyarray(geom.pix_y, dtype=np.float64).value

        #pix_x = event.inst.pixel_pos[telescope_id][1]
        #pix_y = event.inst.pixel_pos[telescope_id][0]

        print(pix_x)
        print(pix_y)

        #print(dl1_camera.pe_samples[dl1_camera.cleaning_mask])
        #print(dl1_camera.time_bin[1][dl1_camera.cleaning_mask])
        #print(pix_x[dl1_camera.cleaning_mask])
        #print(pix_y[dl1_camera.cleaning_mask])

        print('')
        #print(geom)
        print(event.dl0.event_id, event.dl0.run_id, event.mc.energy,
            np.rad2deg(event.mc.alt), np.rad2deg(event.mc.az),
            event.mc.tel[telescope_id].time_slice)


        f, (ax1, ax2) = plt.subplots(1,2,figsize=(20,9))

        #fig = plt.figure(figsize=(10, 9))
        ax1.scatter(pix_x, pix_y, c='grey')
        z1_plot=ax1.scatter(pix_x[dl1_camera.cleaning_mask],
            pix_y[dl1_camera.cleaning_mask],
            c=dl1_camera.time_bin[1][dl1_camera.cleaning_mask])

        #z1_plot=ax1.scatter(pix_x , pix_y, c=dl1_camera.time_bin[1])
        plt.colorbar(z1_plot,ax=ax1)

        #fig = plt.figure(figsize=(10, 9))
        ax2.scatter(pix_x, pix_y, c='grey')
        z2_plot=ax2.scatter(pix_x[dl1_camera.cleaning_mask],
            pix_y[dl1_camera.cleaning_mask],
            c=dl1_camera.pe_samples[dl1_camera.cleaning_mask])
        #z2_plot=ax2.scatter(pix_x , pix_y, c=dl1_camera.pe_samples)
        plt.colorbar(z2_plot,ax=ax2)

        '''
        fig = plt.figure(figsize=(10, 9))
        z2_plot=plt.scatter(pix_x , pix_y,
            c=event.r1.tel[telescope_id].adc_samples[:,16])
        plt.colorbar(z2_plot)

        fig = plt.figure(figsize=(10, 9))
        z2_plot=plt.scatter(pix_x , pix_y, c=dl1_camera.pe_samples_trace[:,17])
        plt.colorbar(z2_plot)
        '''

        plt.show()
    """
