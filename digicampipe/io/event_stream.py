import os
import re

from astropy.io import fits
import numpy as np

from digicampipe.io import zfits, hdf5
from digicampipe.io.slow_container import fill_slow


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


def get_slow_data_info(file_list):
    if len(file_list) == 0:
        print("ERROR: no slow data file given")
        return
    data_structs={}
    # get basic information from slow data
    # (class, min and max timestamp, data location)
    for file in file_list:
        data_struct = {}
        hdulist = fits.open(file)
        data_struct['hdu'] = hdulist[1]
        ts = data_struct['hdu'].data['timestamp']
        good = ts != 0
        events = np.arange(len(ts))
        data_struct['timestamps'] = ts[good]
        data_struct['events'] = events[good]
        data_struct['ts_min'] = min(data_struct['timestamps'])
        data_struct['ts_max'] = max(data_struct['timestamps'])
        filename = os.path.basename(file)
        m = re.match('(?:slow_)?(?P<class>[\w]+?)_[\d\_]+\.fits',
                     filename)
        class_name = m.group("class")
        if class_name not in data_structs.keys():
            data_structs[class_name]=[data_struct]
        else:
            data_structs[class_name].append(data_struct)
    return data_structs


def get_slow_event(info_struct_list, data_ts):
    ts_min_all = [info_struct['ts_min'] for info_struct in info_struct_list]
    ts_max_all = [info_struct['ts_max'] for info_struct in info_struct_list]
    files = [info_struct['hdu']._file.name for info_struct in info_struct_list]
    ts_min_all = np.array(ts_min_all)
    ts_max_all = np.array(ts_max_all)
    files = np.array(files)
    indexes_slow_file = np.logical_and(
        ts_min_all < data_ts,
        ts_max_all > data_ts
    )
    if np.sum(indexes_slow_file) == 0:
        # print('ERROR: no slow file found !')
        return None, None
    if np.sum(indexes_slow_file) > 1:
        print('ERROR: several slow files found !',
              files[indexes_slow_file])
        return None, None
    index_slow_file = np.argwhere(indexes_slow_file.flatten())[0, 0]
    info_struct = info_struct_list[index_slow_file]
    n_slow_event = len(info_struct['timestamps'])
    #argmax stops at first True
    index_next_slow_event = np.argmax(
        info_struct['timestamps']>data_ts
    )
    if index_next_slow_event == 0:
        print('ERROR: wrong ts_min in get_slow_event()')
        return None, None
    index_slow_event = index_next_slow_event - 1
    slow_event = info_struct['events'][index_slow_event]
    hdu = info_struct['hdu']
    return slow_event, hdu


def add_slow_data(data_stream, slow_file_list):
    slow_info_structs = get_slow_data_info(slow_file_list)
    # now for each events look for the latest slow data
    # with ts_slow<=ts_event
    for event in data_stream:
        if len(event.r0.tels_with_data) == 0:
            print("WARNING: no R0 data in event")
        telescope_id = event.r0.tels_with_data[0]
        data_ts = event.r0.tel[telescope_id].local_camera_clock * 1e-6
        for class_name in slow_info_structs.keys():
            info_struct = slow_info_structs[class_name]
            slow_event, hdu = get_slow_event(info_struct, data_ts)
            if slow_event is None:
                print("no %s data found" %class_name)
                continue
            event = fill_slow(class_name, event, hdu, slow_event)
        yield event
