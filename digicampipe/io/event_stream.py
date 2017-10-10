from digicampipe.io import zfits


def event_stream(file_list, camera_geometry, expert_mode=False, max_events=None, allowed_tels=None):

    for file in file_list:

        data_stream = zfits.zfits_event_source(url=file,
                                               expert_mode=expert_mode,
                                               camera_geometry=camera_geometry,
                                               max_events=max_events,
                                               allowed_tels=allowed_tels)

        for event in data_stream:

            yield event

