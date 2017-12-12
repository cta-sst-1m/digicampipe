"""
Container structures for data that should be read or written to disk
"""

from astropy import units as u
from ctapipe.core import Container, Map
try:
    from ctapipe.core import Field
except ImportError:
    from ctapipe.core import Item as Field
from numpy import array, ndarray
from gzip import open as gzip_open
import pickle
from os.path import isfile
from ctapipe.io.serializer import Serializer
from os import remove

__all__ = ['SlowControlContainer',
           'DriveSystemContainer',
           'SlowDataContainer',
           'InstrumentContainer',
           'R0Container',
           'R0CameraContainer',
           'R1Container',
           'R1CameraContainer',
           'DL0Container',
           'DL0CameraContainer',
           'DL1Container',
           'DL1CameraContainer',
           'CentralTriggerContainer',
           'ReconstructedContainer',
           'ReconstructedShowerContainer',
           'ReconstructedEnergyContainer',
           'ParticleClassificationContainer',
           'DataContainer']

class SlowControlContainer(Container):
    """Storage of slow control data .
    This container is filled only on request by using the ??? function.
    """
    timestamp = Field(int, "timestamp")
    trigger_timestamp = Field(int, "trigger timestamp")
    absolute_time = Field(int, "absolute time")
    local_time = Field(int, "local time")
    opcua_time = Field(int, "OPC UA time")
    crates = Field(array, "Crates (3 of them)")
    crate1_timestamps = Field(array, "Crate1 timestamps (10 of them)")
    crate1_status = Field(array, "Crate1 status (10 of them)")
    crate1_temperature = Field(array, "Crate1 temperature (60 of them)")
    crate2_timestamps = Field(array, "Crate2 timestamps (10 of them)")
    crate2_status = Field(array, "Crate2 status (10 of them)")
    crate2_temperature = Field(array, "Crate2 temperature (60 of them)")
    crate3_timestamps = Field(array, "Crate3 timestamps (10 of them)")
    crate3_status = Field(array, "Crate3 status (10 of them)")
    crate3_temperature = Field(array, "Crate3 temperature (60 of them)")
    cst_switches = Field(array, "CST switches (4 of them)")
    cst_parameters = Field(array, "CST parameters (6 of them)")
    app_status = Field(array, "app status ??? (8 of them)")
    trigger_status = Field(int, "trigger status")
    triggers_status = Field(array, "triggers status (6 of them)")
    trigger_parameters = Field(array, "trigger parameters (9 of them)")
    fadc_resync = Field(int, "FADC resync")
    fadc_offset = Field(int, "FADC offset")
    trigger_switches = Field(array, "trigger switches (11 of them)")

class SlowDataContainer(Container):
    """Storage of slow data which changes much slower (typicaly every sec.)
    than events.
    This container is filled only on request by using the ??? function.
    """
    slow_control=Field(SlowControlContainer(), "Slow Control")


class SlowControlContainer(Container):
    """Storage of slow control data.
    This container is filled only on request by using the ??? function.
    """
    timestamp = Field(int, "timestamp")
    trigger_timestamp = Field(int, "trigger timestamp")
    absolute_time = Field(int, "absolute time")
    local_time = Field(int, "local time")
    opcua_time = Field(int, "OPC UA time")
    crates = Field(array, "Crates (3 of them)")
    crate1_timestamps = Field(ndarray, "Crate1 timestamps (10 of them)")
    crate1_status = Field(ndarray, "Crate1 status (10 of them)")
    crate1_temperature = Field(ndarray, "Crate1 temperature (60 of them)")
    crate2_timestamps = Field(ndarray, "Crate2 timestamps (10 of them)")
    crate2_status = Field(ndarray, "Crate2 status (10 of them)")
    crate2_temperature = Field(ndarray, "Crate2 temperature (60 of them)")
    crate3_timestamps = Field(ndarray, "Crate3 timestamps (10 of them)")
    crate3_status = Field(ndarray, "Crate3 status (10 of them)")
    crate3_temperature = Field(ndarray, "Crate3 temperature (60 of them)")
    cst_switches = Field(ndarray, "CST switches (4 of them)")
    cst_parameters = Field(ndarray, "CST parameters (6 of them)")
    app_status = Field(ndarray, "app status ??? (8 of them)")
    trigger_status = Field(int, "trigger status")
    # 3: freq of MUON trigger
    triggers_stats = Field(ndarray, "triggers status (6 of them)")
    # trigger parameters:
    # 0: Fake trigger freq (0 - 125 000 000) [Hz]
    # 1: Fake trigger pulse length (0 - 65 535) [4ns]
    # 2: Readout delay (0 - 2046) [4ns]
    # 3: Readout delay back-plane (0 - 1023) [4ns]
    # 4: Readout window length (1 - 92) [4ns]
    # 5: PATCH7 threshold (0 - 2047, trigger 0 - 1784)
    # 6: PATCH19 threshold (0 - 8191, trigger 0 - 4844)
    # 7: MUON threshold
    # 8: Pixel clipping level
    trigger_parameters = Field(ndarray, "trigger parameters (9 of them)")
    # trigger switches:
    # 0: Trigger enable
    # 1: Trigger mode
    # 2: Trigger trace readout enable
    # 3: Generate trigger
    # 4: Generate clear pulse
    # 5: PATCH7 generate trigger now!
    # 6: PATCH7 trigger enable
    # 7: PATCH19 generate trigger now!
    # 8: PATCH19 trigger enable
    # 9: MUON generate trigger now!
    # 10: MUON trigger enable
    trigger_switches = Field(ndarray, "trigger switches (11 of them)")
    fadc_resync = Field(int, "FADC resync")
    fadc_offset = Field(int, "FADC offset")


class MasterSST1MContainer(Container):
    """Storage of data from SST1M master.
    This container is filled only on request by using the ??? function.
    """
    # TODO: Not implemented yet
    """
    name = 'TIME'; format = 'D'
    name = 'TIMESTAMP'; format = 'K'
    name = 'dplc_time'; format = 'D'
    name = 'events_timings'; format = '3J'
    name = 'splc_control'; format = 'J'
    name = 'target'; format = '64A'
    name = 'dplc_velocity'; format = '2D'
    name = 'schedule'; format = '281A'
    name = 'dplc_azel'; format = '2D'
    name = 'target_radec'; format = '2D'
    name = 'dplc_statuses'; format = '9J'
    name = 'dplc_energy'; format = 'J'
    name = 'splc_time'; format = 'D'
    name = 'splc_state'; format = 'J'
    name = 'command'; format = '64A'
    name = 'master_time'; format = 'D'
    name = 'dplc_errors'; format = '7J'
    name = 'splc_errors'; format = '50J'
    name = 'splc_telemetry'; format = '24D'
    name = 'splc_statuses'; format = '34J'
    name = 'dplc_state'; format = 'J'
    """


class DriveSystemContainer(Container):
    """Storage of data from the drive system.
    This container is filled only on request by using the ??? function.
    """
    # time info
    timestamp = Field(int, "timestamp")
    current_time = Field(float, "current time")
    # position info
    current_track_step_pos_az = Field(float, "current track step pos - azimuth")
    current_track_step_pos_el = Field(float, "current track step pos - elevation")
    current_track_step_t = Field(int, "current track step t")
    current_position_az = Field(float, "current position - azimuth")
    current_position_el = Field(float, "current position - elevation")
    current_nominal_position_az = Field(float, "current nominal position - azimuth")
    current_nominal_position_el = Field(float, "current nominal position - elevation")
    current_velocity_az = Field(float, "current velocity - azimuth")
    current_velocity_el = Field(float, "current velocity - elevation")
    current_max_velocity_az = Field(float, "current maximum velocity - azimuth")
    current_max_velocity_el = Field(float, "current maximum velocity - elevation")
    # cache info
    current_cache_size = Field(int, "current cache size")
    has_cache_capacity = Field(int, "has cache capacity")
    # status info
    is_off = Field(bool, "is off ?")
    has_local_mode_requested = Field(bool, "has local mode requested ?")
    has_remote_mode_requested = Field(bool, "has remote mode requested ?")
    is_in_park_position = Field(bool, "is in park position ?")
    is_in_parking_zone = Field(bool, "is in parking zone ?")
    is_in_start_position = Field(bool, "is in start position ?")
    is_moving = Field(bool, "is moving ?")
    is_tracking = Field(bool, "is tracking ?")
    is_on_source = Field(bool, "is on source ?")
    has_id = Field(str, "has ID")
    has_firmware_release = Field(str, "has firmware release")
    # ???
    in__v_rel = Field(float, "in__v_rel")
    in__track_step_pos_az = Field(float, "in__track_step_pos_az")
    in__track_step_pos_el = Field(float, "in__track_step_pos_el")
    in__position_az = Field(float, "in__position_az")
    in__position_el = Field(float, "in__position_el")
    in__t_after = Field(int, "in__t_after")
    in__track_step_t = Field(int, "in__track_step_t")
    # error info
    capacity_exceeded_error_description = Field(str, "capacity exceeded error description")
    capacity_exceeded_error_ec = Field(int, "capacity exceeded error code")
    capacity_exceeded_error_crit_time = Field(int, "capacity exceeded error critical time")
    capacity_exceeded_error_rev = Field(int, "capacity exceeded error rev")
    invalid_argument_error_description = Field(str, "invalid argument error description")
    invalid_argument_error_ec = Field(int, "invalid argument error code")
    invalid_argument_error_crit_time = Field(int, "invalid argument error critical time")
    invalid_argument_error_rev = Field(int, "invalid argument error rev")
    invalid_operation_error_description = Field(str , "invalid operation error description")
    invalid_operation_error_ec = Field(int, "invalid operation error code")
    invalid_operation_error_crit_time = Field(int, "invalid operation error critical time")
    invalid_operation_error_rev = Field(int, "invalid operation error rev")
    no_permission_error_description = Field(str, "no permission error description")
    no_permission_error_ec = Field(int, "no permission error code")
    no_permission_error_crit_time = Field(int, "no permission error critical time")
    no_permission_error_rev = Field(int, "no permission error rev")
    operation_aborted_error_description = Field(str, "operation aborted error description")
    operation_aborted_error_ec = Field(int, "operation aborted error code")
    operation_aborted_error_crit_time = Field(int, "operation aborted error critical time")
    operation_aborted_error_rev = Field(int, "operation aborted rev")
    operation_stopped_error_description = Field(str, "operation stopped error description")
    operation_stopped_error_ec = Field(int, "operation stopped error code")
    operation_stopped_error_crit_time = Field(int, "operation stopped critical time")
    operation_stopped_error_rev = Field(int, "operation stopped error rev")
    recent_error_name = Field(str, "recent error name")
    recent_error_rev = Field(int, "recent error rev")
    system_is_busy_error_description = Field(str, "system is busy error description")
    system_is_busy_error_ec = Field(int, "system is busy error code")
    system_is_busy_error_crit_time = Field(int, "system is busy error critical time")
    system_is_busy_error_rev = Field(int, "system is busy error rev")


class SlowDataContainer(Container):
    """Storage of slow data whith rates much slower (typicaly every sec.) than events.
    This container is filled only on request by using the io.add_slow_data() function.
    """
    slow_control = Field(SlowControlContainer(), "Slow Control")
    drive_system = Field(DriveSystemContainer(), "Drive System")


class InstrumentContainer(Container):
    """Storage of header info that does not change with event. This is a
    temporary hack until the Instrument module and database is fully
    implemented.  Eventually static information like this will not be
    part of the data stream, but be loaded and accessed from
    functions.
    """
    telescope_ids = Field([], "list of IDs of telescopes used in the run")
    num_pixels = Field(Map(int), "map of tel_id to number of pixels in camera")
    num_channels = Field(Map(int), "map of tel_id to number of channels")
    num_samples = Field(Map(int), "map of tel_id to number of samples")
    geom = Field(Map(None), 'map of tel_if to CameraGeometry')
    cam = Field(Map(None), 'map of tel_id to Camera')
    optics = Field(Map(None), 'map of tel_id to CameraOptics')
    cluster_matrix_7 = Field(Map(ndarray), 'map of tel_id of cluster 7 matrix')
    cluster_matrix_19 = Field(Map(ndarray), 'map of tel_id of cluster 19 matrix')
    patch_matrix = Field(Map(ndarray), 'map of tel_id of patch matrix')


class DL1CameraContainer(Container):
    """Storage of output of camera calibrationm e.g the final calibrated
    image in intensity units and other per-event calculated
    calibration information.
    """
    pe_samples = Field(ndarray, "numpy array containing data volume reduced p.e. samples (n_channels x n_pixels)")
    cleaning_mask = Field(ndarray, "mask for clean pixels")
    time_bin = Field(ndarray, "numpy array containing the bin of maximum (n_pixels)")
    pe_samples_trace = Field(ndarray, "numpy array containing data volume reduced p.e. samples (n_channels x n_pixels, n_samples)")
    on_border = Field(bool, "Boolean telling if the shower touches the camera border or not")
    time_spread = Field(float, 'Time elongation of the shower')


class DL1Container(Container):
    """ DL1 Calibrated Camera Images and associated data"""
    tel = Field(Map(DL1CameraContainer), "map of tel_id to DL1CameraContainer")


class R0CameraContainer(Container):
    """
    Storage of raw data from a single telescope
    event_type_1 definition:
        1 = physics
        2 = muon
        3 = flasher
        4 = dark
        8 = clocked trigger
    """
    pixel_flags = Field(ndarray, 'numpy array containing pixel flags')
    adc_samples = Field(ndarray, "numpy array containing ADC samples (n_channels x n_pixels, n_samples)")
    num_samples = Field(int, "number of time samples for telescope")
    num_pixels = Field(int, "number of pixels in camera")
    baseline = Field(ndarray, "number of time samples for telescope")
    digicam_baseline = Field(ndarray, 'Baseline computed by DigiCam')
    standard_deviation = Field(ndarray, "number of time samples for telescope")
    dark_baseline = Field(ndarray, 'dark baseline')
    hv_off_baseline = Field(ndarray, 'HV off baseline')
    camera_event_id = Field(int, 'Camera event number')
    camera_event_number = Field(int, "camera event number")
    local_camera_clock = Field(float, "camera timestamp")
    gps_time = Field(float, "gps timestamp")
    white_rabbit_time = Field(float, "precise white rabbit based timestamp")
    # camera_event_type:
    # 1: PATCH7 trigger
    # 2: PATCH19 trigger
    # 4: MUON trigger
    # 8: clocked trigger
    # 16: external
    camera_event_type = Field(int, "camera event type")
    array_event_type = Field(int, "array event type")
    trigger_input_traces = Field(ndarray, "trigger patch trace (n_patches)")
    trigger_input_offline = Field(ndarray, "trigger patch trace (n_patches)")
    trigger_output_patch7 = Field(ndarray, "trigger 7 patch cluster trace (n_clusters)")
    trigger_output_patch19 = Field(ndarray, "trigger 19 patch cluster trace (n_clusters)")
    trigger_input_7 = Field(ndarray, 'trigger input CLUSTER7')
    trigger_input_19 = Field(ndarray, 'trigger input CLUSTER19')


class R0Container(Container):
    """
    Storage of a Merged Raw Data Event
    """
    run_id = Field(-1, "run id number")
    event_id = Field(-1, "event id number")
    tels_with_data = Field([], "list of telescopes with data")
    tel = Field(Map(R0CameraContainer), "map of tel_id to R0CameraContainer")


class R1CameraContainer(Container):
    """
    Storage of r1 calibrated data from a single telescope
    """
    adc_samples = Field(ndarray, "baseline subtracted ADCs, (n_pixels, n_samples)")
    nsb = Field(ndarray, "nsb rate in GHz")
    pde = Field(ndarray, "Photo Detection Efficiency at given NSB")
    gain_drop = Field(ndarray, "gain drop")


class R1Container(Container):
    """
    Storage of a r1 calibrated Data Event
    """
    tels_with_data = Field([], "list of telescopes with data")
    tel = Field(Map(R1CameraContainer), "map of tel_id to R1CameraContainer")


class DL0CameraContainer(Container):
    """
    Storage of data volume reduced dl0 data from a single telescope
    """


class DL0Container(Container):
    """
    Storage of a data volume reduced Event
    """

    tels_with_data = Field([], "list of telescopes with data")
    tel = Field(Map(DL0CameraContainer), "map of tel_id to DL0CameraContainer")


class ReconstructedShowerContainer(Container):
    """
    Standard output of algorithms reconstructing shower geometry
    """

    alt = Field(0.0, "reconstructed altitude", unit=u.deg)
    alt_uncert = Field(0.0, "reconstructed altitude uncertainty", unit=u.deg)
    az = Field(0.0, "reconstructed azimuth", unit=u.deg)
    az_uncertainty = Field(0.0, 'reconstructed azimuth uncertainty', unit=u.deg)
    core_x = Field(0.0, 'reconstructed x coordinate of the core position',
                  unit=u.m)
    core_y = Field(0.0, 'reconstructed y coordinate of the core position',
                  unit=u.m)
    core_uncertainty = Field(0.0, 'uncertainty of the reconstructed core position',
                       unit=u.m)
    h_max = Field(0.0, 'reconstructed height of the shower maximum')
    h_max_uncertainty = Field(0.0, 'uncertainty of h_max')
    is_valid = (False, ('direction validity flag. True if the shower direction'
                        'was properly reconstructed by the algorithm'))
    tel_ids = Field([], ('list of the telescope ids used in the'
                        ' reconstruction of the shower'))
    average_size = Field(0.0, 'average size of used')
    goodness_of_fit = Field(0.0, 'measure of algorithm success (if fit)')


class ReconstructedEnergyContainer(Container):
    """
    Standard output of algorithms estimating energy
    """
    energy = Field(-1.0, 'reconstructed energy', unit=u.TeV)
    energy_uncertainty = Field(-1.0, 'reconstructed energy uncertainty', unit=u.TeV)
    is_valid = Field(False, ('energy reconstruction validity flag. True if '
                            'the energy was properly reconstructed by the '
                            'algorithm'))
    goodness_of_fit = Field(0.0, 'goodness of the algorithm fit')


class ParticleClassificationContainer(Container):
    """
    Standard output of gamma/hadron classification algorithms
    """
    prediction = Field(0.0, ('prediction of the classifier, defined between '
                            '[0,1], where values close to 0 are more '
                            'gamma-like, and values close to 1 more '
                            'hadron-like'))
    is_valid = Field(False, ('classificator validity flag. True if the '
                            'predition was successful within the algorithm '
                            'validity range'))

    goodness_of_fit = Field(0.0, 'goodness of the algorithm fit')


class ReconstructedContainer(Container):
    """ collect reconstructed shower info from multiple algorithms """

    shower = Field(Map(ReconstructedShowerContainer),
                  "Map of algorithm name to shower info")
    energy = Field(Map(ReconstructedEnergyContainer),
                  "Map of algorithm name to energy info")
    classification = Field(Map(ParticleClassificationContainer),
                          "Map of algorithm name to classification info")


class DataContainer(Container):
    """ Top-level container for all event information """
    r0 = Field(R0Container(), "Raw Data")
    r1 = Field(R1Container(), "R1 Calibrated Data")
    dl0 = Field(DL0Container(), "DL0 Data Volume Reduced Data")
    dl1 = Field(DL1Container(), "DL1 Calibrated image")
    dl2 = Field(ReconstructedContainer(), "Reconstructed Shower Information")
    inst = Field(InstrumentContainer(), "Instrumental Information (deprecated)")
    slow_data = Field(SlowDataContainer(), "Slow Data Information")


def load_from_pickle_gz(file):
    file = gzip_open(file, "rb")
    while True:
        try:
            yield pickle.load(file)
        except (EOFError, pickle.UnpicklingError):
            return


def save_to_pickle_gz(event_stream, file, overwrite=False, max_events=None):
    if isfile(file):
        if overwrite:
            print('remove old', file, 'file')
            remove(file)
        else:
            print(file, 'exist, exiting...')
            return
    writer = Serializer(filename=file, format='pickle', mode='w')
    counter_events = 0
    for event in event_stream:
        writer.add_container(event)
        counter_events += 1
        if max_events is not None and counter_events >= max_events:
            break
    writer.close()
