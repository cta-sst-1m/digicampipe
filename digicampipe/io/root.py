from pkg_resources import resource_filename
import os
import numpy as np

from ROOT import TFile
from digicampipe.io.containers import DataContainer
from digicampipe.utils import geometry
from cts_core.camera import Camera


def root_event_source(url, camera_geometry, camera, max_events=None,
                      allowed_tels=None):
    n_pixel = 1296
    f = TFile(url)
    camera_events = f.Get("Events/T0")
    mc_events = f.Get("Events/tSimulatedEvents")
    # patch_matrix = geometry.compute_patch_matrix(camera=camera)
    # cluster_7_matrix = geometry.compute_cluster_matrix_7(camera=camera)
    # cluster_19_matrix = geometry.compute_cluster_matrix_19(camera=camera)

    data = DataContainer()
    loaded_telescopes = []
    for event_counter, (event, mc_event) in enumerate(zip(camera_events, mc_events)):
        if max_events is not None and event_counter > max_events:
            break
        tel_id = 1
        data.r0.tels_with_data = [tel_id, ]
        """
        data.r0.tels_with_data = [event.telescope_id, ]
        # remove forbidden telescopes
        if allowed_tels:
            data.r0.tels_with_data = [
                list(filter(lambda x: x in data.r0.tels_with_data, sublist))
                for sublist in allowed_tels
            ]
        """
        for tel_id in data.r0.tels_with_data:
            if tel_id not in loaded_telescopes:
                data.inst.num_channels[tel_id] = 1
                data.inst.num_pixels[tel_id] = n_pixel
                data.inst.geom[tel_id] = camera_geometry
                # data.inst.cluster_matrix_7[tel_id] = cluster_7_matrix
                # data.inst.cluster_matrix_19[tel_id] = cluster_19_matrix
                # data.inst.patch_matrix[tel_id] = patch_matrix
                data.inst.num_samples[tel_id] = event.vFADCTraces0.size()
                loaded_telescopes.append(tel_id)

            r0 = data.r0.tel[tel_id]
            r0.camera_event_number = mc_event.eventNumber
            r0.pixel_flags = None
            r0.local_camera_clock = None
            r0.gps_time = None
            r0.camera_event_type = None
            r0.array_event_type = None
            pixel_samples = np.zeros([n_pixel, data.inst.num_samples[tel_id],], dtype=np.int)
            for p in range(n_pixel):
                pixel_samples[p, :] = getattr(event, 'vFADCTraces'+str(p))
            r0.adc_samples = pixel_samples
            r0.baseline = np.ones([n_pixel,]) * 20
            r0.digicam_baseline = np.ones([n_pixel,]) * 20
            r0.standard_deviation = np.ones([n_pixel,])
        yield data


def main():
    root_file = resource_filename(
        'digicampipe',
        os.path.join(
            'tests',
            'resources',
            'MC_imen_20180404.root'
        )
    )
    digicam_config_file = resource_filename(
        'digicampipe',
        os.path.join(
            'tests',
            'resources',
            'camera_config.cfg'
        )
    )
    digicam = Camera(_config_file=digicam_config_file)
    camera_geometry = geometry.generate_geometry_from_camera(
        camera=digicam
    )
    data_stream = root_event_source(root_file, camera_geometry, digicam)


if __name__ == '__main__':
    main()
