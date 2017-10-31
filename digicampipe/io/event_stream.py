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


from astropy.io import fits
import numpy as np


def add_slow_data(event_stream,slowcontrol_file_list):
    slow_control_structs=[]
    #get basic information from slow data (min and max timestamp, data location)
    for file in slowcontrol_file_list:
        slow_control = {}
        hdulist = fits.open(file)
        nslow_event=hdulist[1].data['timestamp'].shape[0]
        first_slow_event = 0
        last_slow_event = nslow_event - 1
        while hdulist[1].data['timestamp'][first_slow_event] == 0:
            first_slow_event += 1
            if first_slow_event == last_slow_event:
                break
        slow_control['ts_min'] = hdulist[1].data['timestamp'][first_slow_event]
        if first_slow_event == last_slow_event:
            slow_control['ts_max'] = slow_control['ts_min']
        else:
            while hdulist[1].data['timestamp'][last_slow_event] == 0:
                last_slow_event -= 1
                if last_slow_event == 0:
                    break
            slow_control['ts_max'] = hdulist[1].data['timestamp'][last_slow_event]
        slow_control['hdu'] = hdulist[1]
        slow_control['timestamps'] = []
        slow_control['events'] = []
        slow_control_structs.append(slow_control)
    # now for each events look for the lastest slowdata with ts_slow<=ts_event
    index_slow_file = 0
    index_slow_event = 0
    for event in event_stream:
        if len(event.r0.tels_with_data) == 0:
            continue
        telescope_id=event.r0.tels_with_data[0]
        data_ts=event.r0.tel[telescope_id].local_camera_clock*1e-6
        while not (slow_control_structs[index_slow_file]['ts_min']<data_ts and slow_control_structs[index_slow_file]['ts_max']>data_ts):
            index_slow_file+=1
            index_slow_event = 0
            if index_slow_file == len(slow_control_structs):
                break
        if index_slow_file == len(slow_control_structs):
            print("WARNING: slow data file not found")
            yield event
        else:
            # "lazy" get of the the timestamps in slowdata
            if len(slow_control_structs[index_slow_file]['timestamps']) == 0:
                ts=slow_control_structs[index_slow_file]['hdu'].data['timestamp']
                good = ts != 0
                events =np.arange(len(ts))
                slow_control_structs[index_slow_file]['timestamps']=ts[good]
                slow_control_structs[index_slow_file]['events']=events[good]
                nevent=len(slow_control_structs[index_slow_file]['events'])
            # look for the last slow data with a timestamp <= event ts
            ts=slow_control_structs[index_slow_file]['timestamps'][index_slow_event:]
            while (index_slow_event < nevent - 1) and \
                    (slow_control_structs[index_slow_file]['timestamps'][index_slow_event + 1] <= data_ts):
                index_slow_event += 1
            slow_event=slow_control_structs[index_slow_file]['events'][index_slow_event]
            hdu=slow_control_structs[index_slow_file]['hdu']
            # fill container
            event.slowdata.slow_control.timestamp = hdu.data['timestamp'][slow_event]
            event.slowdata.slow_control.trigger_timestamp = hdu.data['trigger_timestamp'][slow_event]
            event.slowdata.slow_control.absolute_time = hdu.data['AbsoluteTime'][slow_event]
            event.slowdata.slow_control.local_time = hdu.data['LocalTime'][slow_event]
            event.slowdata.slow_control.opcua_time = hdu.data['opcuaTime'][slow_event]
            event.slowdata.slow_control.crates = hdu.data['Crates'][slow_event]
            event.slowdata.slow_control.crate1_timestamps = hdu.data['Crate1_timestamps'][slow_event]
            event.slowdata.slow_control.crate1_status = hdu.data['Crate1_status'][slow_event]
            event.slowdata.slow_control.crate1_temperature = hdu.data['Crate1_T'][slow_event]
            event.slowdata.slow_control.crate2_timestamps = hdu.data['Crate2_timestamps'][slow_event]
            event.slowdata.slow_control.crate2_status = hdu.data['Crate2_status'][slow_event]
            event.slowdata.slow_control.crate2_temperature =  hdu.data['Crate2_T'][slow_event]
            event.slowdata.slow_control.crate3_timestamps =  hdu.data['Crate3_timestamps'][slow_event]
            event.slowdata.slow_control.crate3_status =  hdu.data['Crate3_status'][slow_event]
            event.slowdata.slow_control.crate3_temperature =  hdu.data['Crate3_T'][slow_event]
            event.slowdata.slow_control.cst_switches =  hdu.data['cstSwitches'][slow_event]
            event.slowdata.slow_control.cst_parameters =  hdu.data['cstParameters'][slow_event]
            yield event
