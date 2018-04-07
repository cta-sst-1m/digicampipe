from digicampipe.io import zfits, hdf5
import digicampipe.io.hessio_digicam as hsm     #


def event_stream(file_list, camera_geometry, camera, expert_mode=False, max_events=None, mc=False):
    counter = 0
    for file in file_list:
        if not mc:
            data_stream = zfits.zfits_event_source(url=file,
                                                   expert_mode=expert_mode,
                                                   camera_geometry=camera_geometry,
                                                   max_events=max_events,
                                                   camera=camera)
        else:
            """
            data_stream = hdf5.digicamtoy_event_source(url=file,
                                                       camera_geometry=camera_geometry,
                                                       camera=camera,
                                                       max_events=max_events)
            """
            data_stream = hsm.hessio_event_source(file,camera_geometry=camera_geometry, camera=camera, max_events=max_events)    #
        for event in data_stream:
            yield event
            counter += 1
            if max_events and counter>=max_events:
                return
