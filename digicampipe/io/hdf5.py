import h5py
import numpy as np
from tqdm import tqdm

from digicampipe.instrument.camera import DigiCam
from digicampipe.io.containers import DataContainer

__all__ = ['digicamtoy_event_source']


def digicamtoy_event_source(
        url,
        camera=DigiCam,
        max_events=None,
        chunk_size=150,
):
    """A generator that streams data from an HDF5 data file from DigicamToy
    Parameters
    ----------
    url : str
        path to file to open
    max_events : int, optional
        maximum number of events to read
    camera : utils.Camera() default: utils.DigiCam
    chunk_size : Number of events to load into the memory at once
    """

    data = DataContainer()
    hdf5 = h5py.File(url, 'r')

    full_data_set = hdf5['data']['adc_count']
    n_events, n_pixels, n_samples = full_data_set.shape

    if max_events is None:
        max_events = n_events

    max_events = min(max_events, n_events)

    for event_id in tqdm(range(max_events), desc='Events'):

        data.r0.event_id = event_id
        data.r0.tels_with_data = [1, ]

        for tel_id in data.r0.tels_with_data:

            if event_id == 0:
                data.inst.num_channels[tel_id] = 1
                data.inst.num_pixels[tel_id] = n_pixels
                data.inst.geom[tel_id] = camera.geometry
                data.inst.cluster_matrix_7[tel_id] = camera.cluster_7_matrix
                data.inst.cluster_matrix_19[tel_id] = camera.cluster_19_matrix
                data.inst.patch_matrix[tel_id] = camera.patch_matrix
                data.inst.num_samples[tel_id] = n_samples

            if (event_id % chunk_size) == 0:
                index_in_chunk = 0
                chunk_start = (event_id // chunk_size) * chunk_size
                chunk_end = chunk_start + chunk_size
                chunk_end = min(chunk_end, n_events)
                adc_count = full_data_set[chunk_start:chunk_end]

            data.r0.tel[tel_id].camera_event_number = event_id
            data.r0.tel[tel_id].local_camera_clock = None
            data.r0.tel[tel_id].gps_time = event_id
            data.r0.tel[tel_id].camera_event_type = None
            data.r0.tel[tel_id].array_event_type = None
            data.r0.tel[tel_id].adc_samples = adc_count[index_in_chunk]
            baseline = np.ones(
                data.r0.tel[tel_id].adc_samples.shape[:-1]) * np.nan
            data.r0.tel[tel_id].digicam_baseline = baseline
            index_in_chunk += 1

        yield data
