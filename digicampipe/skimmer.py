import numpy as np
from digicampipe.utils import DigiCam


def compute_discrimination_variable(r0_container, patch_coordinates):

    patch_x, patch_y = patch_coordinates
    trigger_time = r0_container.local_camera_clock
    output_trigger_patch7 = np.array(
        list(r0_container.trigger_output_patch7.values()))
    time_above_threshold_per_patch = np.sum(output_trigger_patch7, axis=1)
    max_time_above_threshold = np.max(time_above_threshold_per_patch)
    total_time_above_threshold = np.sum(time_above_threshold_per_patch)
    n_patches_above_threshold = np.sum((time_above_threshold_per_patch > 0))
    patches_id_above_threshold = np.where(
        (time_above_threshold_per_patch > 0))[0]

    list_no = [391, 392, 403, 404, 405, 416, 417]
    nimportecomment = (np.sum(output_trigger_patch7[list_no]) > 0.5)

    sigma_x = np.std(
        (patch_x*time_above_threshold_per_patch)[patches_id_above_threshold])
    # sigma_x = np.std(patch_x[patches_id_above_threshold])
    sigma_y = np.std(
        (patch_y*time_above_threshold_per_patch)[patches_id_above_threshold])
    # sigma_y = np.std(patch_y[patches_id_above_threshold])
    sigma = np.sqrt(sigma_x**2 + sigma_y**2)
    sigma = sigma if not np.isnan(sigma) else 0.

    return (
        trigger_time,
        total_time_above_threshold,
        max_time_above_threshold,
        n_patches_above_threshold,
        sigma,
        nimportecomment)


def skim_events(event_stream):
    discrimination_variable = {
        'time_trigger': [],
        'time_total': [],
        'time_max': [],
        'n_patches': [],
        'shower_spread': []
    }
    patch_x = np.array([
        DigiCam.Patches[i].Vertices[0][0]
        for i in range(len(DigiCam.Patches))
    ])
    patch_y = np.array([
        DigiCam.Patches[i].Vertices[1][0]
        for i in range(len(DigiCam.Patches))
    ])
    patch_coordinates = [patch_x, patch_y]

    for event_number, event in enumerate(event_stream):

        if event_number % 100 == 0:
            print(event_number)

        for telescope_id in event.r0.tels_with_data:

            r0_container = event.r0.tel[telescope_id]
            (
                trigger_time,
                total_time_above_threshold,
                max_time_above_threshold,
                n_patches_above_threshold,
                sigma
            ) = compute_discrimination_variable(
                r0_container,
                patch_coordinates
            )

            discrimination_variable['time_trigger'].append(trigger_time)
            discrimination_variable['time_total'].append(total_time_above_threshold)
            discrimination_variable['time_max'].append(max_time_above_threshold)
            discrimination_variable['n_patches'].append(n_patches_above_threshold)
            discrimination_variable['shower_spread'].append(sigma)

    for key, val in discrimination_variable.items():
        discrimination_variable[key] = np.array(val)

    return discrimination_variable


def compute_patch_coordinates():
    patch_x = np.array([
        DigiCam.Patches[i].Vertices[0][0]
        for i in range(len(DigiCam.Patches))
    ])
    patch_y = np.array([
        DigiCam.Patches[i].Vertices[1][0]
        for i in range(len(DigiCam.Patches))
    ])
    patch_coordinates = [patch_x, patch_y]

    return patch_coordinates
