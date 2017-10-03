from digicampipe.utils import utils, calib
import numpy as np
from digicampipe.utils import geometry
import cts_core.camera as camera
from digicampipe.image import cleaning


def calibrate_to_dl1(event_stream, time_integration_options, additional_mask=None, cleaning_threshold=3):

    for i, event in enumerate(event_stream):

        for telescope_id in event.r0.tels_with_data:
            r0_camera = event.r0.tel[telescope_id]
            r1_camera = event.r1.tel[telescope_id]
            dl1_camera = event.dl1.tel[telescope_id]

            adc_samples = np.array(list(r1_camera.adc_samples.values()))
            gain_init = calib.get_gains()
            gain = gain_init * r1_camera.gain_drop

            # mask pixels which goes above N sigma
            mask_for_cleaning = adc_samples > cleaning_threshold * r0_camera.standard_deviation.reshape(-1, 1)
            dl1_camera.cleaning_mask = np.any(mask_for_cleaning, axis=-1)

            if additional_mask is not None:

                dl1_camera.cleaning_mask = dl1_camera.cleaning_mask * additional_mask

            # Integrate the data
            adc_integrated = utils.integrate(adc_samples, time_integration_options['window_width'])

            pe_samples_trace = adc_integrated / gain[:, np.newaxis] #gain.reshape(-1,1)#[:, np.newaxis]
            n_samples = adc_samples.shape[-1]
            dl1_camera.pe_samples_trace = np.pad(pe_samples_trace, ((0,0), (0, n_samples - pe_samples_trace.shape[-1] % n_samples)), 'constant')

            # Compute the charge
            dl1_camera.pe_samples, dl1_camera.time_bin = utils.extract_charge(adc_integrated,
                                                                              time_integration_options['mask'],
                                                                              time_integration_options['mask_edges'],
                                                                        time_integration_options['peak'],
                                                                              time_integration_options['window_start'],
                                                                              time_integration_options['threshold_saturation'])
            dl1_camera.pe_samples = dl1_camera.pe_samples / gain

            # dl1_camera.time_bin = np.array([dl1_camera.time_bin]) * 4 + r0_camera.local_camera_clock

        yield event


def calibrate_to_dl1_better_cleaning(event_stream, time_integration_options, camera_config_file, picture_threshold=7,
                                     boundary_threshold=4, additional_mask=None):

    cam = camera.Camera(_config_file=camera_config_file)
    geom = geometry.generate_geometry(camera=cam)

    for i, event in enumerate(event_stream):

        for telescope_id in event.r0.tels_with_data:
            r1_camera = event.r1.tel[telescope_id]
            dl1_camera = event.dl1.tel[telescope_id]

            adc_samples = np.array(list(r1_camera.adc_samples.values()))
            gain_init = calib.get_gains()
            gain = gain_init # * r1_camera.gain_drop

            # Integrate the data
            adc_integrated = utils.integrate(adc_samples, time_integration_options['window_width'])

            pe_samples_trace = adc_integrated / gain[:, np.newaxis]
            n_samples = adc_samples.shape[-1]
            dl1_camera.pe_samples_trace = np.pad(pe_samples_trace, ((0,0), (0, n_samples - pe_samples_trace.shape[-1] % n_samples)), 'constant')

            # Compute the charge
            dl1_camera.pe_samples, dl1_camera.time_bin = utils.extract_charge(adc_integrated,
                                                                              time_integration_options['mask'],
                                                                              time_integration_options['mask_edges'],
                                                                              time_integration_options['peak'],
                                                                              time_integration_options['window_start'],
                                                                              time_integration_options['threshold_saturation'])
            dl1_camera.pe_samples = dl1_camera.pe_samples / gain

            # mask pixels which goes above N sigma

            dl1_camera.cleaning_mask = cleaning.tailcuts_clean(geom=geom, image=dl1_camera.pe_samples,
                                picture_threshold=picture_threshold, boundary_threshold=boundary_threshold,
                                keep_isolated_pixels=False)

            dl1_camera.cleaning_mask = cleaning.dilate(geom=geom, mask=dl1_camera.cleaning_mask)
            dl1_camera.cleaning_mask = cleaning.dilate(geom=geom, mask=dl1_camera.cleaning_mask)
            dl1_camera.cleaning_mask = cleaning.dilate(geom=geom, mask=dl1_camera.cleaning_mask)

            if additional_mask is not None:
                dl1_camera.cleaning_mask = dl1_camera.cleaning_mask * additional_mask

            # dl1_camera.time_bin = np.array([dl1_camera.time_bin]) * 4 + r0_camera.local_camera_clock

        yield event