import pkg_resources
from os import path

from cts_core import camera
from digicampipe.utils import geometry


class Camera(camera.Camera):
    ''' cts_core.camera.Camera with ctapipe.instrument.camera.CameraGeometry

    Helper class, for easier construction and use of
    `cts_core.camera.Camera`.

    It can be constucted without a path, it will then use the
    file digicampipe/test/resources/camera_config.cfg,
    which is delivered with digicampipe.

    It does exactly the same as cts_core.camera.Camera but it has
    an *additional* member: `.geometry`
    of type: ctapipe.instrument.camera.CameraGeometry
    which is created immediateyl on construction.
    '''
    def __init__(self, *args, **kwargs):
        if not args and kwargs.get('_config_file', None) is None:
            kwargs['_config_file'] = pkg_resources.resource_filename(
                'digicampipe',
                path.join(
                    'tests',
                    'resources',
                    'camera_config.cfg'
                )
            )
        super().__init__(*args, **kwargs)

        self.geometry = geometry.generate_geometry_from_camera(camera=self)
        self.patch_matrix = geometry.compute_patch_matrix(camera=self)
        self.cluster_7_matrix = geometry.compute_cluster_matrix_7(camera=self)
        self.cluster_19_matrix = geometry.compute_cluster_matrix_19(
            camera=self)


DigiCam = Camera()
