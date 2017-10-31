from digicampipe.io import zfits, hdf5


def event_stream(file_list, camera_geometry, camera, expert_mode=False, max_events=None, mc=False):
    for file in file_list:
        if not mc:
            data_stream = zfits.zfits_event_source(url=file,
                                                   expert_mode=expert_mode,
                                                   camera_geometry=camera_geometry,
                                                   max_events=max_events,
                                                   camera=camera)
        else:
            data_stream = hdf5.digicamtoy_event_source(url=file,
                                                       camera_geometry=camera_geometry,
                                                       camera=camera,
                                                       max_events=max_events)
        for event in data_stream:
            yield event

"""
def add_slow_data(event_stream,slowcontrol_file_list):

    slow_control_structs=[]
    slow_control={}
    for file in slowcontrol_file_list:
        slow_control['ts_min']=0
        slow_control['ts_max']=1
        slow_control_structs.append(slow_control)
        hdulist = fits.open('DigicamSlowControl_20171030_011.fits')
    for event in event_stream:
"""
