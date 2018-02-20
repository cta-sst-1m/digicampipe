import numpy as np
from digicampipe.utils import Processor


class SetPixelsToZero(Processor):
    def __init__(self, unwanted_pixels):
        self.unwanted_pixels = unwanted_pixels

    def __call__(self, event):
        for telescope_id in event.r0.tels_with_data:
            r0_camera = event.r0.tel[telescope_id]
            r0_camera.adc_samples[self.unwanted_pixels] = 0
        return event


class FilterEventTypes(Processor):
    def __init__(self, flags=(0)):
        self.flags = flags

    def __call__(self, event):
        for telescope_id in event.r0.tels_with_data:
            flag = event.r0.tel[telescope_id].camera_event_type

            if flag in self.flags:
                return event
            else:
                return None


class FilterMissingBaseline(Processor):
    def __call__(self, event):
        for telescope_id in event.r0.tels_with_data:
            r0_camera = event.r0.tel[telescope_id]
            condition = np.all(np.isnan(r0_camera.baseline))
            if condition:
                return None
            else:
                return event
