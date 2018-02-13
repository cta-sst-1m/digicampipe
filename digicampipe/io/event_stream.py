from digicampipe.io import zfits, hdf5
from .auxservice import AuxService
from collections import namedtuple


def event_stream(file_list, camera_geometry, camera,
                 expert_mode=False, max_events=None, mc=False):
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


def add_slow_data(
    data_stream,
    aux_services=[
        'DigicamSlowControl',
        'MasterSST1M',
        'PDPSlowControl',
        'SafetyPLC',
        'DriveSystem',
    ],
    basepath=None
):
    services = {
        name: AuxService(name, basepath)
        for name in aux_services
    }

    SlowDataContainer = namedtuple('SlowDataContainer', aux_services)

    for event_id, event in enumerate(data_stream):
        event.slow_data = SlowDataContainer(**{
            name: service.at(event.r0.tel[1].local_camera_clock)
            for (name, service) in services.items()
        })

        yield event
