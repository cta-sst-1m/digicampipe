#!/usr/bin/env python
'''

Example:
  ./simtel_pipeline.py \
  --outfile_path=./ \
  --outfile_suffix=gamma_ze00_az000 \
  --camera_config=./digicampipe/tests/resources/camera_config.cfg \
  --picture_threshold=15 \
  --boundary_threshold=7 \
  --baseline0=9 \
  --baseline1=15 \
  ../sst1m_simulations/*simtel.gz

Usage:
  simtel_pipeline.py [options] <files>...


Options:
  -h --help     Show this screen.
  -o <path>, --outfile_path=<path>  path to the output files
  -s <name>, --outfile_suffix=<name>    suffix of the output files
  -c <path>, --camera_config=<path> camera config file to load Camera()
  -i <int>, --picture_threshold     [default: 15]
  -b <int>, --boundary_threshold    [default: 7]
  -d <int>, --baseline0             [default: 9]
  -e <int>, --baseline1             [default: 15]
  --min_photon <int>     Filtering on big showers [default: 50]
  --display     Display rather than output data
'''
import astropy.units as u
import matplotlib.pyplot as plt
import mc_shower
import numpy as np
from cts_core.camera import Camera
from digicampipe.calib.camera import dl2, r1, dl1
from docopt import docopt

from digicampipe.calib import filter
from digicampipe.instrument import geometry
from digicampipe.io.event_stream import event_stream
from digicampipe.io.save_hillas import save_hillas_parameters
from digicampipe.utils import events_image
from digicampipe.utils import utils
from digicampipe.visualization import EventViewer
from jakub import simtel_baseline


def main(
    files,
    outfile_path,
    outfile_suffix,
    picture_threshold=15,
    boundary_threshold=7,
    baseline0=9,
    baseline1=15,
    ):

    # Input/Output files
    digicam_config_file = args['--camera_config']

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
                                'window_width': 7,  # length of integration window
                                'threshold_saturation': np.inf,
                                'n_samples': 50,
                                'timing_width': 6,
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
    shower_distance = 200 * u.mm

    # Filtering on big showers
    min_photon = 50

    # Config for Hillas parameters analysis
    n_showers = 100000000
    reclean = True

    # Define the event stream
    data_stream = event_stream(
        files,
        camera_geometry=digicam_geometry,
        camera=digicam
    )

    # Clean pixels
    data_stream = filter.set_pixels_to_zero(
        data_stream, unwanted_pixels=pixel_not_wanted)

    # Computation of baseline
    #
    # Methods:
    #
    # simtel_baseline.baseline_data()
    # Baseline is computed as a mean of 'n_bins0' first time samples,
    # 'n_bins1' last time samples.
    # A key assumption in this method is that the shower in simulated
    # data is somewhere in the middle of 50 samples.
    # Each pixel in each event has its own baseline
    #
    # simtel_baseline.baseline_simtel()
    # Baseline is taken as a value reported by sim_telarray
    # event.mc.tel[tel_id].pedestal/event.r0.tel[tel_id].num_samples.
    # That should be OK for all sim_telarray version from April
    # 2018, where an error for DC coupled simulations was corrected.

    # data_stream = simtel_baseline.baseline_data(
    #         data_stream,
    #         n_bins0=baseline0,
    #         n_bins1=baseline1
    # )

    data_stream = simtel_baseline.baseline_simtel(data_stream)

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

    # Save cleaned events - pixels and corresponding values
    filename_pix = 'pixels.txt'
    filename_eventsimage = 'events_image_' + outfile_suffix + '.txt'
    data_stream = events_image.save_events(
        data_stream,
        outfile_path + filename_pix,
        outfile_path + filename_eventsimage
    )

    # Save simulated shower paramters
    filename_showerparam = 'pipedmc_param_' + outfile_suffix + '.txt'
    data_stream = mc_shower.save_shower(
        data_stream,
        outfile_path + filename_showerparam
    )

    # Run the dl2 calibration (Hillas)
    data_stream = dl2.calibrate_to_dl2(
        data_stream, reclean=reclean,
        shower_distance=shower_distance)

    # Save arrival times of photons in pixels passed cleaning
    filename_timing = 'timing_' + outfile_suffix + '.txt'
    data_stream = events_image.save_timing(
        data_stream,
        outfile_path + filename_timing
    )

    # Save mean baseline in event pixels
    filename_baseline = (
                         'baseline_' + outfile_suffix +
                         '_bas' + str(baseline0).zfill(2) +
                         str(baseline1).zfill(2) + '.txt'
                         )
    # data_stream = simtel_baseline.save_mean_event_baseline(
    #    data_stream, directory + filename_baseline)

    if args['--display']:

        with plt.style.context('ggplot'):
            display = EventViewer(data_stream)
        display.draw()

    else:
        # Save Hillas
        hillas_filename = 'hillas_' + outfile_suffix
        save_hillas_parameters(
            data_stream=data_stream,
            n_showers=n_showers,
            output_filename=outfile_path + hillas_filename)
        # save_hillas_parameters_in_text(
        #    data_stream=data_stream,
        #    output_filename=outfile_path + hillas_filename)

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


if __name__ == '__main__':

    args = docopt(__doc__)
    main(
        files=args['<files>'],
        picture_threshold=int(args['--picture_threshold']),
        boundary_threshold=int(args['--boundary_threshold']),
        baseline0=int(args['--baseline0']),
        baseline1=int(args['--baseline1']),
        outfile_suffix=args['--outfile_suffix'],
        outfile_path=args['--outfile_path'],
    )
