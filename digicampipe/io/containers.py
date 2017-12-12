"""
Container structures for data that should be read or written to disk
"""

from astropy import units as u
from astropy.time import Time
from ctapipe.core import Container, Map
try:
    from ctapipe.core import Field
except ImportError:
    from ctapipe.core import Item as Field
from numpy import array,ndarray
from gzip import open as gzip_open
import pickle
from os.path import isfile
from ctapipe.io.serializer import Serializer
from os import remove

__all__ = ['SlowControlContainer'
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
    inst = Field(InstrumentContainer(), "instrumental information (deprecated)")
    slowdata = Field(SlowDataContainer(), "Slow Data Information")

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
