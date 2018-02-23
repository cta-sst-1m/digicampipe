from digicampipe.utils import utils, calib
import numpy as np
from ctapipe.image import cleaning
from . import Processor


class CalibrateToDL1(Processor):
    def __init__(
        self,
        time_integration_options,
        picture_threshold=7,
        boundary_threshold=4,
    ):

        if (
            'peak' not in time_integration_options or
            time_integration_options['peak'] is None
        ):
            (
                time_integration_options['peak'],
                time_integration_options['mask'],
                time_integration_options['mask_edges']
            ) = utils.generate_timing_mask(
                time_integration_options['window_start'],
                time_integration_options['window_width'],
                utils.fake_timing_hist(
                    time_integration_options['n_samples'],
                    time_integration_options['timing_width'],
                    time_integration_options['central_sample']
                )
            )

        self.time_integration_options = time_integration_options
        self.picture_threshold = picture_threshold
        self.boundary_threshold = boundary_threshold

    def __call__(self, event):
        for telescope_id in event.r0.tels_with_data:

            geom = event.inst.geom[telescope_id]
            pixel_id = np.arange(geom.pix_x.shape[0])

            r0_camera = event.r0.tel[telescope_id]
            r1_camera = event.r1.tel[telescope_id]
            dl1_camera = event.dl1.tel[telescope_id]

            adc_samples = r1_camera.adc_samples

            mask_for_cleaning = adc_samples > 3 * r0_camera.standard_deviation[:, np.newaxis]
            dl1_camera.cleaning_mask = np.any(mask_for_cleaning, axis=-1)

            adc_samples[~dl1_camera.cleaning_mask] = 0.

            gain_init = calib.get_gains()
            gain = gain_init

            # Integrate the data
            adc_integrated = utils.integrate(
                adc_samples,
                self.time_integration_options['window_width']
            )

            pe_samples_trace = adc_integrated / gain[:, np.newaxis]
            n_samples = adc_samples.shape[-1]
            dl1_camera.pe_samples_trace = np.pad(
                pe_samples_trace,
                (
                    (0, 0),
                    (0, n_samples - pe_samples_trace.shape[-1] % n_samples)
                ),
                'constant'
            )

            # Compute the charge
            dl1_camera.pe_samples, dl1_camera.time_bin = utils.extract_charge(
                adc_integrated,
                self.time_integration_options['mask'],
                self.time_integration_options['mask_edges'],
                self.time_integration_options['peak'],
                self.time_integration_options['window_start'],
                self.time_integration_options['threshold_saturation']
            )
            dl1_camera.pe_samples = dl1_camera.pe_samples / gain

            # mask pixels which goes above N sigma

            dl1_camera.cleaning_mask *= cleaning.tailcuts_clean(
                geom=geom,
                image=dl1_camera.pe_samples,
                picture_thresh=self.picture_threshold,
                boundary_thresh=self.boundary_threshold,
                keep_isolated_pixels=False)

            # recursive selection of neighboring pixels
            # threshold is 2*boundary_threshold, maybe we should introduce
            # yet a 3rd threshold in the args of the function
            image = dl1_camera.pe_samples
            recursion = True
            border = False
            while recursion:
                recursion = False
                for i in pixel_id[dl1_camera.cleaning_mask]:
                    num_neighbors = 0
                    for j in (
                        pixel_id[geom.neighbor_matrix[i] &
                        ~dl1_camera.cleaning_mask]
                    ):
                        num_neighbors = num_neighbors + 1
                        if image[j] > self.boundary_threshold:
                            dl1_camera.cleaning_mask[j] = True
                            recursion = True
                    if num_neighbors != 6:
                        border = True

            dl1_camera.on_border = border

            dl1_camera.cleaning_mask = cleaning.dilate(
                geom=geom,
                mask=dl1_camera.cleaning_mask)

            if r0_camera.additional_mask is not None:
                dl1_camera.cleaning_mask *= r0_camera.additional_mask

            weight = dl1_camera.pe_samples
            dl1_camera.time_spread = np.average(
                dl1_camera.time_bin[1] * 4,
                weights=weight)
            dl1_camera.time_spread = np.average(
                (dl1_camera.time_bin[1] * 4 - dl1_camera.time_spread)**2,
                weights=weight)
            dl1_camera.time_spread = np.sqrt(dl1_camera.time_spread)

        return event
