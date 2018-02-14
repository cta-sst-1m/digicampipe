from digicampipe.io import zfits, hdf5


def event_stream(
    filelist_or_url,
    camera=None,
    camera_geometry=None,
    expert_mode=None,
    max_events=None,
    mc=False
):
    if isinstance(filelist_or_url, (str, bytes)):
        filelist = [filelist_or_url]
    else:
        filelist = filelist_or_url

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
