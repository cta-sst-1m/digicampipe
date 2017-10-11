"""
Container structures for data that should be read or written to disk
"""

from astropy import units as u
from astropy.time import Time
from ctapipe.core import Container, Item, Map
from numpy import ndarray
import numpy as np

__all__ = ['InstrumentContainer',
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


class InstrumentContainer(Container):
    """Storage of header info that does not change with event. This is a
    temporary hack until the Instrument module and database is fully
    implemented.  Eventually static information like this will not be
    part of the data stream, but be loaded and accessed from
    functions.
    """

    telescope_ids = Item([], "list of IDs of telescopes used in the run")
    num_pixels = Item(Map(int), "map of tel_id to number of pixels in camera")
    num_channels = Item(Map(int), "map of tel_id to number of channels")
    num_samples = Item(Map(int), "map of tel_id to number of samples")
    geom = Item(Map(None), 'map of tel_if to CameraGeometry')
    cam = Item(Map(None), 'map of tel_id to Camera')
    optics = Item(Map(None), 'map of tel_id to CameraOptics')


class DL1CameraContainer(Container):
    """Storage of output of camera calibrationm e.g the final calibrated
    image in intensity units and other per-event calculated
    calibration information.
    """

    pe_samples = Item(None, ("numpy array containing data volume reduced "
                             "p.e. samples"
                             "(n_channels x n_pixels)"))
    cleaning_mask = Item(None, "mask for clean pixels")
    time_bin = Item(None, ("numpy array containing the bin of maximum"
                           "(n_pixels)"))

    pe_samples_trace = Item(None, ("numpy array containing data volume reduced "
                             "p.e. samples"
                             "(n_channels x n_pixels, n_samples)"))

    on_border = Item(None, ("Boolean telling if the shower touches the camera border or not "
                            "none"
                            "none"))

class DL1Container(Container):
    """ DL1 Calibrated Camera Images and associated data"""
    tel = Item(Map(DL1CameraContainer), "map of tel_id to DL1CameraContainer")


class R0CameraContainer(Container):
    """
    Storage of raw data from a single telescope
    """
    adc_sums = Item(None, ("numpy array containing integrated ADC data "
                           "(n_channels x n_pixels)"))
    adc_samples = Item(None, ("numpy array containing ADC samples"
                              "(n_channels x n_pixels, n_samples)"))
    num_samples = Item(None, "number of time samples for telescope")

    num_pixels = Item(None, "number of pixels in camera")

    baseline = Item(None, "number of time samples for telescope")

    standard_deviation = Item(None, "number of time samples for telescope")

    dark_baseline = Item(ndarray, 'dark baseline')

    hv_off_baseline = Item(None, 'HV off baseline')

    camera_event_id = Item(-1, 'Camera event number')

    camera_event_number = Item(-1, "camera event number")

    local_camera_clock = Item(-1, "camera timestamp")

    gps_time = Item(-1, "gps timestamp")

    white_rabbit_time = Item(-1, "precise white rabbit based timestamp")

    event_type_1 = Item(-1, "event type (1)")

    event_type_2 = Item(-1, "event Type (2)")

    trigger_input_traces = Item(ndarray, ("trigger patch trace", "(n_patches)"))

    trigger_output_patch7 = Item(ndarray, ("trigger 7 patch cluster trace", "(n_clusters)"))

    trigger_output_patch19 = Item(ndarray, ("trigger 19 patch cluster trace", "(n_clusters)"))


class R0Container(Container):
    """
    Storage of a Merged Raw Data Event
    """

    run_id = Item(-1, "run id number")
    event_id = Item(-1, "event id number")
    tels_with_data = Item([], "list of telescopes with data")
    tel = Item(Map(R0CameraContainer), "map of tel_id to R0CameraContainer")


class R1CameraContainer(Container):
    """
    Storage of r1 calibrated data from a single telescope
    """

    adc_samples = Item(ndarray, "baseline subtracted ADCs, (n_pixels, n_samples)")
    nsb = Item(ndarray, "nsb rate in GHz")
    gain_drop = Item(ndarray, "gain drop")


class R1Container(Container):
    """
    Storage of a r1 calibrated Data Event
    """

    tels_with_data = Item([], "list of telescopes with data")
    tel = Item(Map(R1CameraContainer), "map of tel_id to R1CameraContainer")


class DL0CameraContainer(Container):
    """
    Storage of data volume reduced dl0 data from a single telescope
    """


class DL0Container(Container):
    """
    Storage of a data volume reduced Event
    """

    tels_with_data = Item([], "list of telescopes with data")
    tel = Item(Map(DL0CameraContainer), "map of tel_id to DL0CameraContainer")


class ReconstructedShowerContainer(Container):
    """
    Standard output of algorithms reconstructing shower geometry
    """

    alt = Item(0.0, "reconstructed altitude", unit=u.deg)
    alt_uncert = Item(0.0, "reconstructed altitude uncertainty", unit=u.deg)
    az = Item(0.0, "reconstructed azimuth", unit=u.deg)
    az_uncertainty = Item(0.0, 'reconstructed azimuth uncertainty', unit=u.deg)
    core_x = Item(0.0, 'reconstructed x coordinate of the core position',
                  unit=u.m)
    core_y = Item(0.0, 'reconstructed y coordinate of the core position',
                  unit=u.m)
    core_uncertainty = Item(0.0, 'uncertainty of the reconstructed core position',
                       unit=u.m)
    h_max = Item(0.0, 'reconstructed height of the shower maximum')
    h_max_uncertainty = Item(0.0, 'uncertainty of h_max')
    is_valid = (False, ('direction validity flag. True if the shower direction'
                        'was properly reconstructed by the algorithm'))
    tel_ids = Item([], ('list of the telescope ids used in the'
                        ' reconstruction of the shower'))
    average_size = Item(0.0, 'average size of used')
    goodness_of_fit = Item(0.0, 'measure of algorithm success (if fit)')


class ReconstructedEnergyContainer(Container):
    """
    Standard output of algorithms estimating energy
    """
    energy = Item(-1.0, 'reconstructed energy', unit=u.TeV)
    energy_uncertainty = Item(-1.0, 'reconstructed energy uncertainty', unit=u.TeV)
    is_valid = Item(False, ('energy reconstruction validity flag. True if '
                            'the energy was properly reconstructed by the '
                            'algorithm'))
    goodness_of_fit = Item(0.0, 'goodness of the algorithm fit')


class ParticleClassificationContainer(Container):
    """
    Standard output of gamma/hadron classification algorithms
    """
    prediction = Item(0.0, ('prediction of the classifier, defined between '
                            '[0,1], where values close to 0 are more '
                            'gamma-like, and values close to 1 more '
                            'hadron-like'))
    is_valid = Item(False, ('classificator validity flag. True if the '
                            'predition was successful within the algorithm '
                            'validity range'))

    goodness_of_fit = Item(0.0, 'goodness of the algorithm fit')


class ReconstructedContainer(Container):
    """ collect reconstructed shower info from multiple algorithms """

    shower = Item(Map(ReconstructedShowerContainer),
                  "Map of algorithm name to shower info")
    energy = Item(Map(ReconstructedEnergyContainer),
                  "Map of algorithm name to energy info")
    classification = Item(Map(ParticleClassificationContainer),
                          "Map of algorithm name to classification info")


class DataContainer(Container):
    """ Top-level container for all event information """
    r0 = Item(R0Container(), "Raw Data")
    r1 = Item(R1Container(), "R1 Calibrated Data")
    dl0 = Item(DL0Container(), "DL0 Data Volume Reduced Data")
    dl1 = Item(DL1Container(), "DL1 Calibrated image")
    dl2 = Item(ReconstructedContainer(), "Reconstructed Shower Information")
    inst = Item(InstrumentContainer(), "instrumental information (deprecated")
