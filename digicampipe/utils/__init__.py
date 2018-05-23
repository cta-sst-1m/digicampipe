import pkg_resources
from os import path
from scipy.sparse import csr_matrix

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
            self.config_file = kwargs['_config_file']
        elif args:
            self.config_file = args[0]
        else:
            self.config_file = kwargs['_config_file']
        super().__init__(*args, **kwargs)

        geometry_kwargs = {}
        if 'source_x' in kwargs:
            geometry_kwargs['source_x'] = kwargs['source_x']
        if 'source_y' in kwargs:
            geometry_kwargs['source_y'] = kwargs['source_y']
        self.geometry = geometry.generate_geometry_from_camera(
            camera=self,
            **geometry_kwargs
        )
        self.patch_matrix = geometry.compute_patch_matrix(camera=self)
        self.patch_matrix = csr_matrix(self.patch_matrix)
        self.cluster_7_matrix = geometry.compute_cluster_matrix_7(camera=self)
        self.cluster_7_matrix = csr_matrix(self.cluster_7_matrix)
        self.cluster_19_matrix = geometry.compute_cluster_matrix_19(
            camera=self)
        self.cluster_19_matrix = csr_matrix(self.cluster_19_matrix)


DigiCam = Camera()
