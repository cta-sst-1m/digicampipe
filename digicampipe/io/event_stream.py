from digicampipe.io import zfits, hdf5


def event_stream(
    filelist,
    camera=None,
    camera_geometry=None,
    expert_mode=None,
    max_events=None,
    mc=False
):
    # If the caller gives us a path and not a list of paths,
    # we convert it to a list.
    # This is not clean but convenient.
    if isinstance(filelist, (str, bytes)):
        filelist = [filelist]

    if mc:
        source = hdf5.digicamtoy_event_source
    else:
        source = zfits.zfits_event_source

    for file in filelist:
        data_stream = source(
            url=file,
            camera=camera,
            camera_geometry=camera_geometry,
            max_events=max_events,
        )
        for event in data_stream:
            yield event
