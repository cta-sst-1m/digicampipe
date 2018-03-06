"""
Container structures for data that should be read or written to disk. The main data container is DataContainer()
and holds the containers of each data processing level. The data processing levels start from R0 up to DL2,
where R0 holds the cameras raw data and DL2 the air shower high-level parameters.
In general each major pipeline step is associated with a given data level. Please keep in mind that the data level definition and the associated fields might change rapidly
as there is no final data level definition.
"""

from astropy import units as u
from astropy.time import Time
from ctapipe.core import Container, Map
from ctapipe.instrument import SubarrayDescription
try:
    from ctapipe.core import Field
except ImportError:
    from ctapipe.core import Item as Field
from numpy import ndarray
from gzip import open as gzip_open
import pickle
from os.path import isfile
from ctapipe.io.serializer import Serializer
from os import remove

__all__ = ['InstrumentContainer',
           'R0Container',
           'R0CameraContainer',
           'R1Container',
           'R1CameraContainer',
           'DL0Container',
           'DL0CameraContainer',
           'DL1Container',
           'DL1CameraContainer',
           'MCEventContainer',
           'MCHeaderContainer',
           'MCCameraEventContainer',
           'CentralTriggerContainer',
           'ReconstructedContainer',
           'ReconstructedShowerContainer',
           'ReconstructedEnergyContainer',
           'ParticleClassificationContainer',
           'DataContainer']


class InstrumentContainer(Container):
    """Storage of header info that does not change with event. This is a
    temporary hack until the Instrument module and database is fully
    implemented.  Eventually static information like this will not be
    part of the data stream, but be loaded and accessed from
    functions.
    """

    subarray = Field(SubarrayDescription("MonteCarloArray"),
                     "SubarrayDescription from the instrument module")
    optical_foclen = Field(Map(ndarray), "map of tel_id to focal length")
    tel_pos = Field(Map(ndarray), "map of tel_id to telescope position")
    pixel_pos = Field(Map(ndarray), "map of tel_id to pixel positions")
    telescope_ids = Field([], "list of IDs of telescopes used in the run")
    num_pixels = Field(Map(int), "map of tel_id to number of pixels in camera")
    num_channels = Field(Map(int), "map of tel_id to number of channels")
    num_samples = Field(Map(int), "map of tel_id to number of samples")
    geom = Field(Map(None), 'map of tel_if to CameraGeometry')
    cam = Field(Map(None), 'map of tel_id to Camera')
    optics = Field(Map(None), 'map of tel_id to CameraOptics')
    cluster_matrix_7 = Field(Map(ndarray), 'map of tel_id of cluster 7 matrix')
    cluster_matrix_19 = Field(
                              Map(ndarray),
                              'map of tel_id of cluster 19 matrix'
                             )
    patch_matrix = Field(Map(ndarray), 'map of tel_id of patch matrix')
    mirror_dish_area = Field(Map(float),
                             "map of tel_id to the area of the mirror dish",
                             unit=u.m**2)
    mirror_numtiles = Field(Map(int),
                            "map of tel_id to the number of \
                            tiles for the mirror")


class DL1CameraContainer(Container):
    """Storage of output of camera calibrationm e.g the final calibrated
    image in intensity units and other per-event calculated
    calibration information.
    """

    pe_samples = Field(ndarray, "numpy array containing data volume reduced \
                       p.e. samples (n_channels x n_pixels)")
    cleaning_mask = Field(ndarray, "mask for clean pixels")
    time_bin = Field(ndarray, "numpy array containing the bin of maximum \
                    (n_pixels)")
    pe_samples_trace = Field(ndarray, "numpy array containing data volume \
                             reduced p.e. samples (n_channels x n_pixels, \
                             n_samples)")
    on_border = Field(bool, "Boolean telling if the shower touches the camera \
                      border or not")
    time_spread = Field(float, 'Time elongation of the shower')


class DL1Container(Container):
    """ DL1 Calibrated Camera Images and associated data"""
    tel = Field(Map(DL1CameraContainer), "map of tel_id to DL1CameraContainer")


class R0CameraContainer(Container):
    """
    Storage of raw data from a single telescope

    :param pixel_flags: a ndarray to flag the pixels
    :type pixel_flags: ndarray (n_pixels, ) (bool)
    :param adc_samples: a ndarray (n_pixels, n_samples) containing
                        the waveforms in each pixel
    :type adc_samples: ndarray (n_pixels, n_samples, ) (uint16)
    :param adc_sums: numpy array containing integrated ADC data
                     (n_channels, x n_pixels)
    :type adc_sums: ndarray (n_channels, x n_pixels)
    :param baseline: baseline holder for baseline computation using
                     clocked triggers
    :type baseline: ndarray (n_pixels, ) (float)
    :param digicam_baseline: baseline computed by DigiCam of pre-samples
                             (using 1024 samples)
    :type digicam_baseline: ndarray (n_pixels, ) (uint16)
    :param standard_deviation: baseline standard deviation holder for baseline
                               computed using clocked triggers
    :type standard deviation: ndarray (n_pixels, ) (float)
    :param dark_baseline: baseline holder for baseline computed in dark
                          condition (lid closed)
    :type dark_baseline: ndarray (n_pixels, ) (float)
    :param hv_off_baseline: baseline computed with sensors just bellow
                            breakdown voltage (or without bias voltage applied)
    :type hv_off_baseline: ndarray (n_pixels, ) (float)
    :param camera_event_id: unique event identification provided by DigiCam
    :type camera_event_id: (int)
    :param camera_event_number: event number within the first trigger of
                                operation
    :type camera_event_number: (int)
    :param local_camera_clock: time stamp from internal DigiCam clock (ns)
    :type local_camera_clock: (float)
    :param gps_time: time stamp provided by a precise external clock
                     (synchronized between hardware components)

    """
    pixel_flags = Field(ndarray, 'numpy array containing pixel flags')
    adc_samples = Field(ndarray,
                        "numpy array containing ADC samples"
                        "(n_channels x n_pixels, n_samples)")
    adc_sums = Field(ndarray, "numpy array containing integrated ADC data"
                     "(n_channels, x n_pixels)")
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
    trigger_output_patch7 = Field(ndarray, "trigger 7 patch cluster trace \
                                  (n_clusters)")
    trigger_output_patch19 = Field(ndarray, "trigger 19 patch cluster trace \
                                   (n_clusters)")
    trigger_input_7 = Field(ndarray, 'trigger input CLUSTER7')
    trigger_input_19 = Field(ndarray, 'trigger input CLUSTER19')
    num_samples = Field(int, "number of time samples for telescope")


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

    adc_samples = Field(ndarray, "baseline subtracted ADCs, (n_pixels, \
                        n_samples)")
    pulse_indices = Field(list, 'Reconstructed time of each waveform (n_pixels, ?)')
    reconstructed_charge = Field(list, 'Reconstructed charge of each signal in each waveform (n_pixels, ?)')
    nsb = Field(ndarray, "nsb rate in GHz")
    pde = Field(ndarray, "Photo Detection Efficiency at given NSB")
    gain_drop = Field(ndarray, "gain drop")


class R1Container(Container):
    """
    Storage of a r1 calibrated Data Event
    """

    run_id = Field(-1, "run id number")
    event_id = Field(-1, "event id number")
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

    run_id = Field(-1, "run id number")
    event_id = Field(-1, "event id number")
    tels_with_data = Field([], "list of telescopes with data")
    tel = Field(Map(DL0CameraContainer), "map of tel_id to DL0CameraContainer")


class MCCameraEventContainer(Container):
    """
    Storage of mc data for a single telescope that change per event
    """
    photo_electron_image = Field(
        Map(), "reference image in pure photoelectrons, with no noise"
    )
    reference_pulse_shape = Field(
        None, "reference pulse shape for each channel"
    )
    time_slice = Field(0, "width of time slice", unit=u.ns)
    dc_to_pe = Field(None, "DC/PE calibration arrays from MC file")
    pedestal = Field(None, "pedestal calibration arrays from MC file")
    azimuth_raw = Field(
        0, "Raw azimuth angle [radians from N->E] for the telescope"
    )
    altitude_raw = Field(0, "Raw altitude angle [radians] for the telescope")
    azimuth_cor = Field(
        0, "the tracking Azimuth corrected for pointing errors for \
        the telescope"
    )
    altitude_cor = Field(
        0, "the tracking Altitude corrected for pointing \
        errors for the telescope"
    )


class MCEventContainer(Container):
    """
    Monte-Carlo
    """
    energy = Field(0.0, "Monte-Carlo Energy", unit=u.TeV)
    alt = Field(0.0, "Monte-carlo altitude", unit=u.deg)
    az = Field(0.0, "Monte-Carlo azimuth", unit=u.deg)
    core_x = Field(0.0, "MC core position", unit=u.m)
    core_y = Field(0.0, "MC core position", unit=u.m)
    h_first_int = Field(0.0, "Height of first interaction")
    tel = Field(
        Map(MCCameraEventContainer), "map of tel_id to MCCameraEventContainer"
    )

    mc_event_offset_fov = Field(
                                Map(ndarray),
                                "offset of pointing direction in camera \
                                f.o.v. divided by focal length, i.e. \
                                converted to radians: [0] = Camera x \
                                (downwards in normal pointing, i.e. \
                                increasing Alt) [1] = Camera y -> Az.")


class MCHeaderContainer(Container):
    """
    Monte-Carlo information that doesn't change per event
    """
    run_array_direction = Field([], (
        "the tracking/pointing direction in "
        "[radians]. Depending on 'tracking_mode' "
        "this either contains: "
        "[0]=Azimuth, [1]=Altitude in mode 0, "
        "OR "
        "[0]=R.A., [1]=Declination in mode 1."
    ))


class CentralTriggerContainer(Container):

    gps_time = Field(Time, "central average time stamp")
    tels_with_trigger = Field([], "list of telescopes with data")


class ReconstructedShowerContainer(Container):
    """
    Standard output of algorithms reconstructing shower geometry
    """

    alt = Field(0.0, "reconstructed altitude", unit=u.deg)
    alt_uncert = Field(0.0, "reconstructed altitude uncertainty", unit=u.deg)
    az = Field(0.0, "reconstructed azimuth", unit=u.deg)
    az_uncertainty = Field(0.0, 'reconstructed azimuth uncertainty',
                           unit=u.deg)
    core_x = Field(0.0, 'reconstructed x coordinate of the core position',
                  unit=u.m)
    core_y = Field(0.0, 'reconstructed y coordinate of the core position',
                  unit=u.m)
    core_uncertainty = Field(0.0, 'uncertainty of the reconstructed core \
                             position', unit=u.m)
    h_max = Field(0.0, 'reconstructed height of the shower maximum')
    h_max_uncertainty = Field(0.0, 'uncertainty of h_max')
    is_valid = (False,
                ('direction validity flag. True if the shower direction'
                 'was properly reconstructed by the algorithm'))
    tel_ids = Field([],
                    ('list of the telescope ids used in the'
                     ' reconstruction of the shower'))
    average_size = Field(0.0, 'average size of used')
    goodness_of_fit = Field(0.0, 'measure of algorithm success (if fit)')


class ReconstructedEnergyContainer(Container):
    """
    Standard output of algorithms estimating energy
    """
    energy = Field(-1.0, 'reconstructed energy', unit=u.TeV)
    energy_uncertainty = Field(
                               -1.0, 'reconstructed energy uncertainty',
                               unit=u.TeV
                               )
    is_valid = Field(False,
                     ('energy reconstruction validity flag. True if '
                      'the energy was properly reconstructed by the '
                      'algorithm'))
    goodness_of_fit = Field(0.0, 'goodness of the algorithm fit')


class ParticleClassificationContainer(Container):
    """
    Standard output of gamma/hadron classification algorithms
    """
    prediction = Field(0.0,
                       ('prediction of the classifier, defined between '
                        '[0,1], where values close to 0 are more '
                        'gamma-like, and values close to 1 more '
                        'hadron-like'))
    is_valid = Field(False,
                     ('classificator validity flag. True if the '
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
    """ Top-level container for all event information.
    Each field is representing a specific data processing level from (R0 to
    DL2) Please keep in mind that the data level definition and the associated
    fields might change rapidly as there is not a final data format. The data
    levels R0, R1, DL1, contains sub-containers for each telescope.
    After DL2 the data is not processed at the telescope level.
    """
    r0 = Field(R0Container(), "Raw Data")
    r1 = Field(R1Container(), "Raw Common Data")
    dl0 = Field(DL0Container(), "DL0 Data Volume Reduced Data")
    dl1 = Field(DL1Container(), "DL1 Calibrated image")
    dl2 = Field(ReconstructedContainer(), "Reconstructed Shower Information")
    mc = Field(MCEventContainer(), "Monte-Carlo data")
    mcheader = Field(MCHeaderContainer(), "Monte-Carlo run header data")
    inst = Field(InstrumentContainer(), "Instrumental information")
    slow_data = Field(None, "Slow Data Information")
    trig = Field(CentralTriggerContainer(), "central trigger information")
    count = Field(0, "number of events processed")


class CalibrationContainer(Container):
    """
    This Container() is used for the camera calibration pipeline.
    It is meant to save each step of the calibration pipeline
    """

    adc_samples = Field(ndarray, 'the raw data')
    n_pixels = Field(int, 'number of pixels')
    camera_baseline = Field(ndarray, 'the baseline computed by the camera')
    reconstructed_baseline = Field(ndarray, 'the reconstructed baseline')
    pulse_mask = Field(ndarray, 'mask of adc_samples if True the adc sample'
                                   ' contains a pulse was detected else False')
    reconstructed_amplitude = Field(ndarray, 'array of the same shape as '
                                             'adc_samples giving the'
                                             ' reconstructed pulse amplitude'
                                             ' for each adc sample')

    reconstructed_charge = Field(ndarray, 'array of the same shape as '
                                          'adc_samples giving the '
                                          'reconstructed charge for each adc '
                                          'sample')

    reconstructed_number_of_pe = Field(ndarray, 'estimated number of photon '
                                                'electrons for each adc sample'
                                                )


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
