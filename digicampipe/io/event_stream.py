from digicampipe.io import zfits, hdf5
from .auxservice import AuxService
from collections import namedtuple
import digicampipe.io.hessio_digicam as hsm     #

def event_stream(
    filelist,
    camera=None,
    camera_geometry=None,
    expert_mode=None,
    max_events=None,
    mc='real'
):
    # If the caller gives us a path and not a list of paths,
    # we convert it to a list.
    # This is not clean but convenient.
    if isinstance(filelist, (str, bytes)):
        filelist = [filelist]

    if mc == 'digicamtoy':
        source = hdf5.digicamtoy_event_source
    elif mc == 'real':
        source = zfits.zfits_event_source
    elif mc == 'simtel':
        source = hsm.hessio_event_source
    for file in filelist:
        data_stream = source(
            url=file,
            camera=camera,
            camera_geometry=camera_geometry,
            max_events=max_events,
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
