# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Components to read ZFITS data.
This requires the protozfitsreader python library to be installed
"""
from ctapipe.io.zfits import ZFitsFileReader
from digicampipe.utils import Camera as DigiCam
import warning

__all__ = ['zfits_event_source']


def zfits_event_source(
    url,
    camera=None,
    camera_geometry=None,
    max_events=None,
    allowed_tels=None,
    expert_mode=False
):
    """A generator that streams data from an ZFITs data file
    Parameters
    ----------
    url : str
        path to file to open
    max_events : int, optional
        maximum number of events to read
    allowed_tels : list[int]
        select only a subset of telescope, if None, all are read. This can
        be used for example emulate the final CTA data format, where there
        would be 1 telescope per file (whereas in current monte-carlo,
        they are all interleaved into one file)
    expert_mode : bool
        to read trigger input and output data
    camera_geometry: CameraGeometry()
        camera containing info on pixels modules etc.
    """
    if camera is None:
        camera = DigiCam()

    if camera_geometry is not None:
        warning.warn(
            "camera_geometry is deprecated, "
            "use digicampipe.utils.Camera",
            DeprecationWarning
        )
    else:
        camera_geometry = camera.geometry

    if allowed_tels is not None:
        warning.warn(
            "allowed_tels is deprecated, "
            "I have no idea what you should do :-|",
            DeprecationWarning
        )

    for event in ZFitsFileReader(input_url=url, max_events=max_events):
        for tel_id in event.r0.tels_with_data:
            event.inst.num_channels[tel_id] = event.num_channels
            event.inst.num_pixels[tel_id] = event.n_pixels
            event.inst.geom[tel_id] = camera_geometry
            event.inst.cluster_matrix_7[tel_id] = camera.cluster_7_matrix
            event.inst.cluster_matrix_19[tel_id] = camera.cluster_19_matrix
            event.inst.patch_matrix[tel_id] = camera.patch_matrix
            event.inst.num_samples[tel_id] = event.num_samples

        yield event
