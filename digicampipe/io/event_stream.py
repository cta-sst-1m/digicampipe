from digicampipe.io import zfits, hdf5


def event_stream(
    file_list,
    camera,
    camera_geometry=None,
    expert_mode=None,
    max_events=None,
    mc=False
):

    for file in file_list:

        if not mc:

            data_stream = zfits.zfits_event_source(
                url=file,
                expert_mode=expert_mode,
                camera_geometry=camera_geometry,
                max_events=max_events,
                camera=camera
            )
        else:

            data_stream = hdf5.digicamtoy_event_source(
                url=file,
                camera_geometry=camera_geometry,
                camera=camera,
                max_events=max_events
            )

        for event in data_stream:

            yield event
