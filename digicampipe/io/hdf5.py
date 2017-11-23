from digicampipe.io.containers import DataContainer
import digicampipe.utils as utils
import h5py

__all__ = ['digicamtoy_event_source']


def digicamtoy_event_source(url, camera_geometry, camera, max_events=None):
    """A generator that streams data from an HDF5 data file from DigicamToy
    Parameters
    ----------
    url : str
        path to file to open
    max_events : int, optional
        maximum number of events to read
    camera_geometry: CameraGeometry()
        camera containing info on pixels modules etc.
    camera : cts_core.Camera()
    """

    hdf5 = h5py.File(url, 'r')

    geometry = camera_geometry
    patch_matrix = utils.geometry.compute_patch_matrix(camera=camera)
    cluster_7_matrix = utils.geometry.compute_cluster_matrix_7(camera=camera)
    cluster_19_matrix = utils.geometry.compute_cluster_matrix_19(camera=camera)
    data = DataContainer()
    n_pixels, n_samples, n_events = hdf5['data']['adc_count'].shape

    if max_events is None:

        max_events = n_events

    for event_id in range(max_events):

        data.r0.event_id = event_id
        data.r0.tels_with_data = [1, ]

        for tel_id in data.r0.tels_with_data:

            if event_id < 1:

                data.inst.num_channels[tel_id] = 1
                data.inst.num_pixels[tel_id] = n_pixels
                data.inst.geom[tel_id] = geometry
                data.inst.cluster_matrix_7[tel_id] = cluster_7_matrix
                data.inst.cluster_matrix_19[tel_id] = cluster_19_matrix
                data.inst.patch_matrix[tel_id] = patch_matrix
                data.inst.num_samples[tel_id] = n_samples

            data.r0.tel[tel_id].camera_event_number = event_id
            data.r0.tel[tel_id].local_camera_clock = event_id
            data.r0.tel[tel_id].gps_time = event_id
            data.r0.tel[tel_id].camera_event_type = None
            data.r0.tel[tel_id].array_event_type = None
            data.r0.tel[tel_id].adc_samples = hdf5['data']['adc_count'][..., event_id]

        yield data
