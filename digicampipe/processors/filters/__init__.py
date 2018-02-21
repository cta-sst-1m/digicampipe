import numpy as np
from . import Processor
from . import SkipEvent


class SetPixelsToZero(Processor):
    def __init__(self, unwanted_pixels):
        self.unwanted_pixels = unwanted_pixels

    def __call__(self, event):
        for telescope_id in event.r0.tels_with_data:
            r0_camera = event.r0.tel[telescope_id]
            r0_camera.adc_samples[self.unwanted_pixels] = 0
            additional_mask = np.ones(len(r0_camera.adc_samples), dtype=bool)
            additional_mask[self.unwanted_pixels] = False
            r0_camera.additional_mask = additional_mask
        return event


class FilterEventTypes(Processor):
    def __init__(self, flags=(0)):
        self.flags = flags

    def __call__(self, event):
        for telescope_id in event.r0.tels_with_data:
            flag = event.r0.tel[telescope_id].camera_event_type

            if flag not in self.flags:
                raise SkipEvent

            return event


class FilterMissingBaseline(Processor):
    def __call__(self, event):
        for telescope_id in event.r0.tels_with_data:
            r0_camera = event.r0.tel[telescope_id]
            condition = np.all(np.isnan(r0_camera.baseline))
            if condition:
                raise SkipEvent

            return event


class FilterShower(Processor):
    def __init__(self, min_photon):
        self.min_photon = min_photon

    def __call__(self, event):
        for telescope_id in event.r0.tels_with_data:
            dl1 = event.dl1.tel[telescope_id]
            n_photons = np.sum(dl1.pe_samples[dl1.cleaning_mask])
            if n_photons < self.min_photon:
                raise SkipEvent

            return event
