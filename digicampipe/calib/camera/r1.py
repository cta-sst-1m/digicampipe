import numpy as np
from digicampipe.utils import utils,calib


def calibrate_to_r1(event_stream, calib_container, time_integration_options):
    cleaning_threshold = 3.

    pixel_list = list(range(1296))

    for event in event_stream:
        # Check that the event is physics trigger
        if event.trig.trigger_flag != 0:
            yield event
            continue
        # Check that there were enough random trigger to compute the baseline
        if not calib_container.baseline_ready :
            yield event
            continue

        for telescope_id in event.r0.tels_with_data:
            # Get the R0 and R1 containers
            r0_camera = event.r0.tel[telescope_id]
            r1_camera = event.r1.tel[telescope_id]
            # Get the ADCs
            adc_samples = np.array(list(r0_camera.adc_samples.values()))
            # Get the mean and standard deviation
            r1_camera.pedestal_mean = calib_container.baseline
            r1_camera.pedestal_std = calib_container.std_dev
            # Subtract baseline to the data
            adc_samples = adc_samples - r1_camera.pedestal_mean
            # Compute the gain drop and NSB
            if calib_container.dark_baseline is None :
                # compute NSB and Gain drop from STD
                r1_camera.gain_drop = calib.compute_gain_drop(adc_samples,'std')
                r1_camera.nsb  = calib.compute_nsb_rate(adc_samples,'std')
            else:
                # compute NSB and Gain drop from baseline shift
                r1_camera.gain_drop = calib.compute_gain_drop(adc_samples,'mean')
                r1_camera.nsb  = calib.compute_nsb_rate(adc_samples,'mean')

            gain_init = calib.get_gains()
            gain = gain_init * r1_camera.gain_drop

            # mask pixels which goes above N sigma
            mask_for_cleaning = adcs_samples > cleaning_threshold  * r1_camera.pedestal_std
            mask_for_cleaning = np.any(mask_for_cleaning,axis=-1)

            # Integrate the data
            adc_samples = utils.integrate(adc_samples, time_integration_options['window_width'])

            # Compute the charge
            charge = utils.extract_charge(adc_samples, time_integration_options['mask'],
                                    time_integration_options['mask_edges'],
                                    time_integration_options['peak'],
                                    time_integration_options['window_start'],
                                    time_integration_options['threshold_saturation'])

            r1_camera.pe_samples = dict(zip(pixel_list, charge))

            yield event

