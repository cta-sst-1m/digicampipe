import warnings
from digicampipe.io.containers import DataContainer
import digicampipe.utils as utils
import h5py
import numpy as np

__all__ = ['digicamtoy_event_source']


def digicamtoy_event_source(
    url,
    camera=None,
    max_events=None,
    camera_geometry=None
):
    """A generator that streams data from an HDF5 data file from DigicamToy
    Parameters
    ----------
    url : str
        path to file to open
    max_events : int, optional
        maximum number of events to read
    camera : utils.Camera() or None for DigiCam
    camera_geometry: soon to be deprecated
    """
    if camera is None:
        camera = utils.DigiCam

    if not isinstance(camera, utils.Camera):
        warnings.warn(
            "camera should be utils.Camera not cts_core.camera.Camera",
            FutureWarning
        )

        patch_matrix = utils.geometry.compute_patch_matrix(camera=camera)
        cluster_7_matrix = utils.geometry.compute_cluster_matrix_7(camera=camera)
        cluster_19_matrix = utils.geometry.compute_cluster_matrix_19(camera=camera)
    else:
        patch_matrix = camera.patch_matrix
        cluster_7_matrix = camera.cluster_7_matrix
        cluster_19_matrix = camera.cluster_19_matrix

    data = DataContainer()
    hdf5 = h5py.File(url, 'r')

    data_set = hdf5['data']['adc_count']
    n_events, n_pixels, n_samples = data_set.shape
    adc_count = np.zeros(data_set.shape)
    data_set.read_direct(adc_count)

    if max_events is None:

        max_events = n_events

    max_events = min(max_events, n_events)

    for event_id in range(max_events):

        data.r0.event_id = event_id
        data.r0.tels_with_data = [1, ]

        for tel_id in data.r0.tels_with_data:

            if event_id == 0:

                data.inst.num_channels[tel_id] = 1
                data.inst.num_pixels[tel_id] = n_pixels
                data.inst.geom[tel_id] = camera.geometry
                data.inst.cluster_matrix_7[tel_id] = cluster_7_matrix
                data.inst.cluster_matrix_19[tel_id] = cluster_19_matrix
                data.inst.patch_matrix[tel_id] = patch_matrix
                data.inst.num_samples[tel_id] = n_samples

            data.r0.tel[tel_id].camera_event_number = event_id
            data.r0.tel[tel_id].local_camera_clock = None
            data.r0.tel[tel_id].gps_time = event_id
            data.r0.tel[tel_id].camera_event_type = None
            data.r0.tel[tel_id].array_event_type = None
            data.r0.tel[tel_id].adc_samples = adc_count[event_id]

        yield data
