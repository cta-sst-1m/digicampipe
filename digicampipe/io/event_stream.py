from digicampipe.io import zfits, hdf5
from astropy.io import fits
import numpy as np


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


def add_slow_data(data_stream, slow_control_file_list=None, drive_system_file_list=None):
    if slow_control_file_list is not None:
        data_stream = add_slow_control_data(data_stream, slow_control_file_list)
    if drive_system_file_list is not None:
        data_stream = add_drive_system_data(data_stream, drive_system_file_list)
    for event in data_stream:
        yield event


def get_slow_data_info(file_list):
    if len(file_list) == 0:
        print("ERROR: no slow data file given")
        return
    data_structs=[]
    # get basic information from slow data (min and max timestamp, data location)
    for file in file_list:
        data_struct = {}
        hdulist = fits.open(file)
        nslow_event=hdulist[1].data['timestamp'].shape[0]
        first_slow_event = 0
        last_slow_event = nslow_event - 1
        while hdulist[1].data['timestamp'][first_slow_event] == 0:
            first_slow_event += 1
            if first_slow_event == last_slow_event:
                break
        data_struct['ts_min'] = hdulist[1].data['timestamp'][first_slow_event]
        if first_slow_event == last_slow_event:
            data_struct['ts_max'] = data_struct['ts_min']
        else:
            while hdulist[1].data['timestamp'][last_slow_event] == 0:
                last_slow_event -= 1
                if last_slow_event == 0:
                    break
            data_struct['ts_max'] = hdulist[1].data['timestamp'][last_slow_event]
        data_struct['hdu'] = hdulist[1]
        data_struct['timestamps'] = []
        data_struct['events'] = []
        data_structs.append(data_struct)
    return data_structs


def add_slow_control_data(data_stream, slow_control_file_list):
    slow_control_structs = get_slow_data_info(slow_control_file_list)
    # now for each events look for the latest slow data with ts_slow<=ts_event
    index_slow_file = 0
    index_slow_event = 0
    for event in data_stream:
        if len(event.r0.tels_with_data) == 0:
            print("WARNING: no R0 data in event")
            yield event
        telescope_id = event.r0.tels_with_data[0]
        data_ts = event.r0.tel[telescope_id].local_camera_clock*1e-6
        while slow_control_structs[index_slow_file]['ts_max'] <= data_ts <= slow_control_structs[index_slow_file]['ts_min']:
            index_slow_file += 1
            index_slow_event = 0
            if index_slow_file == len(slow_control_structs):
                break
        if index_slow_file == len(slow_control_structs):
            print("WARNING: slow data file not found")
            yield event
        else:
            if len(slow_control_structs[index_slow_file]['timestamps']) == 0:
                ts = slow_control_structs[index_slow_file]['hdu'].data['timestamp']
                good = ts != 0
                events = np.arange(len(ts))
                slow_control_structs[index_slow_file]['timestamps'] = ts[good]
                slow_control_structs[index_slow_file]['events'] = events[good]
            nevent = len(slow_control_structs[index_slow_file]['events'])
            # look for the last slow data with a timestamp <= event ts
            while (index_slow_event < nevent - 1) and \
                    (slow_control_structs[index_slow_file]['timestamps'][index_slow_event + 1] <= data_ts):
                index_slow_event += 1
            slow_event = slow_control_structs[index_slow_file]['events'][index_slow_event]
            hdu = slow_control_structs[index_slow_file]['hdu']
            # fill container
            event.slow_data.slow_control.timestamp = hdu.data['timestamp'][slow_event]
            event.slow_data.slow_control.trigger_timestamp = hdu.data['trigger_timestamp'][slow_event]
            event.slow_data.slow_control.absolute_time = hdu.data['AbsoluteTime'][slow_event]
            event.slow_data.slow_control.local_time = hdu.data['LocalTime'][slow_event]
            event.slow_data.slow_control.opcua_time = hdu.data['opcuaTime'][slow_event]
            event.slow_data.slow_control.crates = hdu.data['Crates'][slow_event]
            event.slow_data.slow_control.crate1_timestamps = hdu.data['Crate1_timestamps'][slow_event]
            event.slow_data.slow_control.crate1_status = hdu.data['Crate1_status'][slow_event]
            event.slow_data.slow_control.crate1_temperature = hdu.data['Crate1_T'][slow_event]
            event.slow_data.slow_control.crate2_timestamps = hdu.data['Crate2_timestamps'][slow_event]
            event.slow_data.slow_control.crate2_status = hdu.data['Crate2_status'][slow_event]
            event.slow_data.slow_control.crate2_temperature = hdu.data['Crate2_T'][slow_event]
            event.slow_data.slow_control.crate3_timestamps = hdu.data['Crate3_timestamps'][slow_event]
            event.slow_data.slow_control.crate3_status = hdu.data['Crate3_status'][slow_event]
            event.slow_data.slow_control.crate3_temperature = hdu.data['Crate3_T'][slow_event]
            event.slow_data.slow_control.cst_switches = hdu.data['cstSwitches'][slow_event]
            event.slow_data.slow_control.cst_parameters = hdu.data['cstParameters'][slow_event]
            event.slow_data.slow_control.app_status = hdu.data['appStatus'][slow_event]
            event.slow_data.slow_control.trigger_status = hdu.data['trigger_status'][slow_event]
            event.slow_data.slow_control.triggers_stats = hdu.data['triggerStatus'][slow_event]
            event.slow_data.slow_control.trigger_parameters = hdu.data['triggerParameters'][slow_event]
            event.slow_data.slow_control.trigger_switches = hdu.data['triggerSwitches'][slow_event]
            event.slow_data.slow_control.fadc_resync = hdu.data['FadcResync'][slow_event]
            event.slow_data.slow_control.fadc_offset = hdu.data['FadcOffset'][slow_event]
            yield event


def add_drive_system_data(data_stream, drive_system_file_list):
    drive_system_structs = get_slow_data_info(drive_system_file_list)
    # now for each events look for the latest slow data with ts_slow<=ts_event
    index_slow_file = 0
    index_slow_event = 0
    for event in data_stream:
        if len(event.r0.tels_with_data) == 0:
            print("WARNING: no R0 data in event")
            yield event
        telescope_id = event.r0.tels_with_data[0]
        data_ts = event.r0.tel[telescope_id].local_camera_clock*1e-6
        while drive_system_structs[index_slow_file]['ts_max'] <= data_ts <= drive_system_structs[index_slow_file]['ts_min']:
            index_slow_file += 1
            index_slow_event = 0
            if index_slow_file == len(drive_system_structs):
                break
        if index_slow_file == len(drive_system_structs):
            print("WARNING: drive system file not found")
            yield event
        else:
            if len(drive_system_structs[index_slow_file]['timestamps']) == 0:
                ts = drive_system_structs[index_slow_file]['hdu'].data['timestamp']
                good = ts != 0
                events = np.arange(len(ts))
                drive_system_structs[index_slow_file]['timestamps'] = ts[good]
                drive_system_structs[index_slow_file]['events'] = events[good]
            nevent = len(drive_system_structs[index_slow_file]['events'])
            # look for the last slow data with a timestamp <= event ts
            while (index_slow_event < nevent - 1) and \
                    (drive_system_structs[index_slow_file]['timestamps'][index_slow_event + 1] <= data_ts):
                index_slow_event += 1
            slow_event = drive_system_structs[index_slow_file]['events'][index_slow_event]
            hdu = drive_system_structs[index_slow_file]['hdu']
            # fill container
            event.slow_data.drive_system.timestamp = hdu.data['timestamp'][slow_event]
            event.slow_data.drive_system.current_time = hdu.data['current_time'][slow_event]
            event.slow_data.drive_system.current_track_step_pos_az = hdu.data['current_track_step_pos_az'][slow_event]
            event.slow_data.drive_system.current_track_step_pos_el = hdu.data['current_track_step_pos_el'][slow_event]
            event.slow_data.drive_system.current_track_step_t = hdu.data['current_track_step_t'][slow_event]
            event.slow_data.drive_system.current_position_az = hdu.data['current_position_az'][slow_event]
            event.slow_data.drive_system.current_position_el = hdu.data['current_position_el'][slow_event]
            event.slow_data.drive_system.current_nominal_position_az = hdu.data['current_nominal_position_az'][slow_event]
            event.slow_data.drive_system.current_nominal_position_el = hdu.data['current_nominal_position_el'][slow_event]
            event.slow_data.drive_system.current_velocity_az = hdu.data['current_velocity_az'][slow_event]
            event.slow_data.drive_system.current_velocity_el = hdu.data['current_velocity_el'][slow_event]
            event.slow_data.drive_system.current_max_velocity_az = hdu.data['current_max_velocity_az'][slow_event]
            event.slow_data.drive_system.current_max_velocity_el = hdu.data['current_max_velocity_el'][slow_event]
            event.slow_data.drive_system.current_cache_size = hdu.data['current_cache_size'][slow_event]
            event.slow_data.drive_system.has_cache_capacity = hdu.data['has_cache_capacity'][slow_event]
            event.slow_data.drive_system.is_off = hdu.data['is_off'][slow_event]
            event.slow_data.drive_system.has_local_mode_requested = hdu.data['has_local_mode_requested'][slow_event]
            event.slow_data.drive_system.has_remote_mode_requested = hdu.data['has_remote_mode_requested'][slow_event]
            event.slow_data.drive_system.is_in_park_position = hdu.data['is_in_park_position'][slow_event]
            event.slow_data.drive_system.is_in_parking_zone = hdu.data['is_in_parking_zone'][slow_event]
            event.slow_data.drive_system.is_in_start_position = hdu.data['is_in_start_position'][slow_event]
            event.slow_data.drive_system.is_moving = hdu.data['is_moving'][slow_event]
            event.slow_data.drive_system.is_tracking = hdu.data['is_tracking'][slow_event]
            event.slow_data.drive_system.is_on_source = hdu.data['is_on_source'][slow_event]
            event.slow_data.drive_system.has_id = hdu.data['has_id'][slow_event]
            event.slow_data.drive_system.has_firmware_release = hdu.data['has_firmware_release'][slow_event]
            event.slow_data.drive_system.in__v_rel = hdu.data['in__v_rel'][slow_event]
            event.slow_data.drive_system.in__track_step_pos_az = hdu.data['in__track_step_pos_az'][slow_event]
            event.slow_data.drive_system.in__track_step_pos_el = hdu.data['in__track_step_pos_el'][slow_event]
            event.slow_data.drive_system.in__position_az = hdu.data['in__position_az'][slow_event]
            event.slow_data.drive_system.in__position_el = hdu.data['in__position_el'][slow_event]
            event.slow_data.drive_system.in__t_after = hdu.data['in__t_after'][slow_event]
            event.slow_data.drive_system.in__track_step_t = hdu.data['in__track_step_t'][slow_event]
            event.slow_data.drive_system.capacity_exceeded_error_description = hdu.data['capacity_exceeded_error_description'][slow_event]
            event.slow_data.drive_system.capacity_exceeded_error_ec = hdu.data['capacity_exceeded_error_ec'][slow_event]
            event.slow_data.drive_system.capacity_exceeded_error_crit_time = hdu.data['capacity_exceeded_error_crit_time'][slow_event]
            event.slow_data.drive_system.capacity_exceeded_error_rev = hdu.data['capacity_exceeded_error_rev'][slow_event]
            event.slow_data.drive_system.invalid_argument_error_description = hdu.data['invalid_argument_error_description'][slow_event]
            event.slow_data.drive_system.invalid_argument_error_ec = hdu.data['invalid_argument_error_ec'][slow_event]
            event.slow_data.drive_system.invalid_argument_error_crit_time = hdu.data['invalid_argument_error_crit_time'][slow_event]
            event.slow_data.drive_system.invalid_argument_error_rev = hdu.data['invalid_argument_error_rev'][slow_event]
            event.slow_data.drive_system.invalid_operation_error_description = hdu.data['invalid_operation_error_description'][slow_event]
            event.slow_data.drive_system.invalid_operation_error_ec = hdu.data['invalid_operation_error_ec'][slow_event]
            event.slow_data.drive_system.invalid_operation_error_crit_time = hdu.data['invalid_operation_error_crit_time'][slow_event]
            event.slow_data.drive_system.invalid_operation_error_rev = hdu.data['invalid_operation_error_rev'][slow_event]
            event.slow_data.drive_system.no_permission_error_description = hdu.data['no_permission_error_description'][slow_event]
            event.slow_data.drive_system.no_permission_error_ec = hdu.data['no_permission_error_ec'][slow_event]
            event.slow_data.drive_system.no_permission_error_crit_time = hdu.data['no_permission_error_crit_time'][slow_event]
            event.slow_data.drive_system.no_permission_error_rev = hdu.data['no_permission_error_rev'][slow_event]
            event.slow_data.drive_system.operation_aborted_error_description = hdu.data['operation_aborted_error_description'][slow_event]
            event.slow_data.drive_system.operation_aborted_error_ec = hdu.data['operation_aborted_error_ec'][slow_event]
            event.slow_data.drive_system.operation_aborted_error_crit_time = hdu.data['operation_aborted_error_crit_time'][slow_event]
            event.slow_data.drive_system.operation_aborted_error_rev = hdu.data['operation_aborted_error_rev'][slow_event]
            event.slow_data.drive_system.operation_stopped_error_description = hdu.data['operation_stopped_error_description'][slow_event]
            event.slow_data.drive_system.operation_stopped_error_ec = hdu.data['operation_stopped_error_ec'][slow_event]
            event.slow_data.drive_system.operation_stopped_error_crit_time = hdu.data['operation_stopped_error_crit_time'][slow_event]
            event.slow_data.drive_system.operation_stopped_error_rev = hdu.data['operation_stopped_error_rev'][slow_event]
            event.slow_data.drive_system.recent_error_name = hdu.data['recent_error_name'][slow_event]
            event.slow_data.drive_system.recent_error_rev = hdu.data['recent_error_rev'][slow_event]
            event.slow_data.drive_system.system_is_busy_error_description = hdu.data['system_is_busy_error_description'][slow_event]
            event.slow_data.drive_system.system_is_busy_error_ec = hdu.data['system_is_busy_error_ec'][slow_event]
            event.slow_data.drive_system.system_is_busy_error_crit_time = hdu.data['system_is_busy_error_crit_time'][slow_event]
            event.slow_data.drive_system.system_is_busy_error_rev =  hdu.data['system_is_busy_error_rev'][slow_event]
            yield event

