import numpy as np


def save_file(mc_data_all, filename_showerparam):
    np.savetxt(filename_showerparam, mc_data_all, '%1.5f')


def save_shower(event_stream, filename_showerparam):
    mc_data_all = []

    for event in event_stream:
        event_id = event.dl0.event_id
        run_id = event.dl0.run_id
        core_distance = np.sqrt(
            event.mc.core_x ** 2 + event.mc.core_y ** 2).value
        energy = event.mc.energy.value
        theta = 90 - np.rad2deg(event.mc.alt).value
        phi = np.rad2deg(event.mc.az).value
        h_first_int = event.mc.h_first_int.value
        offset_fov_x = event.mc.mc_event_offset_fov[0]
        offset_fov_y = event.mc.mc_event_offset_fov[1]
        core_x = event.mc.core_x.value
        core_y = event.mc.core_y.value

        mc_data = np.hstack((event_id, run_id, core_distance, energy,
                             theta, phi, h_first_int, offset_fov_x,
                             offset_fov_y,
                             core_x, core_y))
        mc_data_all.append(mc_data)

        yield event

    # save mc parameters of showers for all events
    save_file(mc_data_all, filename_showerparam)
