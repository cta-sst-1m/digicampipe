import numpy as np
from cts_core import camera
import astropy.units as u


def stats_events(event_stream):

    variable = {'time_trigger': [], 'time_total': [], 'time_max': [], 'n_patches': [], 'shower_spread': []}

    cts_path = '/home/alispach/Documents/PhD/ctasoft/CTS/'
    digicam = camera.Camera(_config_file=cts_path + 'config/camera_config.cfg')
    patch_x = np.array([digicam.Patches[i].Vertices[0][0] for i in range(len(digicam.Patches))])
    patch_y = np.array([digicam.Patches[i].Vertices[1][0] for i in range(len(digicam.Patches))])

    for event_number, event in enumerate(event_stream):

        if event_number % 100 == 0:
            print(event_number)

        for telescope_id in event.r0.tels_with_data:

            trigger_time = event.r0.tel[telescope_id].local_camera_clock
            output_trigger_patch7 = np.array(list(event.r0.tel[telescope_id].trigger_output_patch7.values()))
            time_above_threshold_per_patch = np.sum(output_trigger_patch7, axis=1)
            max_time_above_threshold = np.max(time_above_threshold_per_patch)
            total_time_above_threshold = np.sum(time_above_threshold_per_patch)
            n_patches_above_threshold = np.sum((time_above_threshold_per_patch > 0))
            patches_id_above_threshold = np.where((time_above_threshold_per_patch > 0))[0]

            sigma_x = np.std((patch_x*time_above_threshold_per_patch)[patches_id_above_threshold])
            # sigma_x = np.std(patch_x[patches_id_above_threshold])
            sigma_y = np.std((patch_y*time_above_threshold_per_patch)[patches_id_above_threshold])
            # sigma_y = np.std(patch_y[patches_id_above_threshold])
            sigma = np.sqrt(sigma_x**2 + sigma_y**2)
            sigma = sigma if not np.isnan(sigma) else 0.


            variable['time_trigger'].append(trigger_time)
            variable['time_total'].append(total_time_above_threshold)
            variable['time_max'].append(max_time_above_threshold)
            variable['n_patches'].append(n_patches_above_threshold)
            variable['shower_spread'].append(sigma)

    for key, val in variable.items():

        variable[key] = np.array(val)

    return variable
