# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Components to read simtel data.

This requires the eventio python library to be installed
"""
import logging
import numpy as np
from tqdm import tqdm

from astropy import units as u
from astropy.coordinates import Angle
from astropy.time import Time
from eventio.simtel.simtelfile import SimTelFile
from ctapipe.io.simteleventsource import SimTelEventSource

from digicampipe.io.containers import DataContainer


logger = logging.getLogger(__name__)


__all__ = [
    'simtel_event_source',
]


def simtel_event_source(url, camera=None, max_events=None,
                        allowed_tels=None, requested_event=None,
                        use_event_id=False, event_id=None, disable_bar=False):
    """A generator that streams data from an EventIO/HESSIO MC data file
    (e.g. a standard CTA data file.)

    Parameters
    ----------
    url : str
        path to file to open
    max_events : int, optional
        maximum number of events to read
    allowed_tels : list[int]
        select only a subset of telescope, if None, all are read. This can
        be used for example emulate the final CTA data format, where there
        would be 1 telescope per file (whereas in current monte-carlo,
        they are all interleaved into one file)
    requested_event : int
        Seek to a paricular event index
    use_event_id : bool
        If True ,'requested_event' now seeks for a particular event id instead
        of index
    disable_bar : Unused, for compatibility with other readers
    """

    if event_id is not None:
        raise ValueError('Event id feature not implemented yet! \n'
                         'Use event_id=None')
    if not SimTelEventSource.is_compatible(url):
        raise ValueError(url, 'is not a valid simtel file')
    data = DataContainer()
    data.meta['origin'] = "hessio"

    # some hessio_event_source specific parameters
    data.meta['input'] = url
    # data.meta['max_events'] = max_events

    with SimTelFile(url) as file:
        telescope_descriptions = file.telescope_descriptions
        subarray_info = SimTelEventSource.prepare_subarray_info(
            telescope_descriptions,
            file.header
        )
        counter = 0
        for array_event in tqdm(file, disable=disable_bar):
            # Seek to requested event
            if requested_event is not None:
                current = counter
                if use_event_id:
                    current = event_id
                if not current == requested_event:
                    counter += 1
                    continue
            run_id = file.header['run']
            event_id = array_event['event_id']
            tels_with_data = set(array_event['telescope_events'].keys())
            data.inst.subarray = subarray_info
            data.r0.run_id = run_id
            data.r0.event_id = event_id
            data.r0.tels_with_data = tels_with_data
            data.r1.run_id = run_id
            data.r1.event_id = event_id
            data.r1.tels_with_data = tels_with_data
            data.dl0.run_id = run_id
            data.dl0.event_id = event_id
            data.dl0.tels_with_data = tels_with_data
            # handle telescope filtering by taking the intersection of
            # tels_with_data and allowed_tels
            if allowed_tels is not None:
                selected = data.r0.tels_with_data & allowed_tels
                if len(selected) == 0:
                    continue  # skip event
                data.r0.tels_with_data = selected
                data.r1.tels_with_data = selected
                data.dl0.tels_with_data = selected
            trigger_information = array_event['trigger_information']
            mc_event = array_event['mc_event']
            mc_shower = array_event['mc_shower']

            data.trig.tels_with_trigger = \
                trigger_information['triggered_telescopes']
            time_s, time_ns = trigger_information['gps_time']
            data.trig.gps_time = Time(time_s * u.s, time_ns * u.ns,
                                      format='unix', scale='utc')
            data.mc.energy = mc_shower['energy'] * u.TeV
            data.mc.alt = Angle(mc_shower['altitude'], u.rad)
            data.mc.az = Angle(mc_shower['azimuth'], u.rad)
            data.mc.core_x = mc_event['xcore'] * u.m
            data.mc.core_y = mc_event['ycore'] * u.m
            first_int = mc_shower['h_first_int'] * u.m
            data.mc.h_first_int = first_int
            data.mc.x_max = mc_shower['xmax'] * u.g / (u.cm ** 2)
            data.mc.shower_primary_id = mc_shower['primary_id']

            # mc run header data
            data.mcheader.run_array_direction = Angle(
                file.header['direction'] * u.rad
            )
            mc_run_head = file.mc_run_headers[-1]
            data.mcheader.corsika_version = mc_run_head['shower_prog_vers']
            data.mcheader.simtel_version = mc_run_head['detector_prog_vers']
            data.mcheader.energy_range_min = mc_run_head['E_range'][0] * u.TeV
            data.mcheader.energy_range_max = mc_run_head['E_range'][1] * u.TeV
            data.mcheader.prod_site_B_total = mc_run_head['B_total'] * u.uT
            data.mcheader.prod_site_B_declination = Angle(
                mc_run_head['B_declination'] * u.rad)
            data.mcheader.prod_site_B_inclination = Angle(
                mc_run_head['B_inclination'] * u.rad)
            data.mcheader.prod_site_alt = mc_run_head['obsheight'] * u.m
            data.mcheader.spectral_index = mc_run_head['spectral_index']

            # this should be done in a nicer way to not re-allocate the
            # data each time (right now it's just deleted and garbage
            # collected)
            data.r0.tel.clear()
            data.r1.tel.clear()
            data.dl0.tel.clear()
            data.dl1.tel.clear()
            data.mc.tel.clear()  # clear the previous telescopes

            telescope_events = array_event['telescope_events']
            tracking_positions = array_event['tracking_positions']
            for tel_id, telescope_event in telescope_events.items():
                telescope_description = telescope_descriptions[tel_id]
                camera_monitorings = array_event['camera_monitorings'][tel_id]
                pedestal = camera_monitorings['pedestal']
                laser_calib = array_event['laser_calibrations'][tel_id]
                data.mc.tel[tel_id].dc_to_pe = laser_calib['calib']
                data.mc.tel[tel_id].pedestal = pedestal
                adc_samples = telescope_event.get('adc_samples')
                n_pixel = adc_samples.shape[-2]
                if adc_samples is None:
                    adc_samples = telescope_event['adc_sums'][:, :, np.newaxis]
                data.r0.tel[tel_id].adc_samples = adc_samples
                data.r0.tel[tel_id].num_samples = adc_samples.shape[-1]
                # We should not calculate stuff in an event source
                # if this is not needed, we calculate it for nothing
                data.r0.tel[tel_id].adc_sums = adc_samples.sum(axis=-1)
                baseline = pedestal / adc_samples.shape[1]
                data.r0.tel[tel_id].digicam_baseline = np.squeeze(baseline)
                data.r0.tel[tel_id].camera_event_number = event_id

                pixel_settings = telescope_description['pixel_settings']
                data.mc.tel[tel_id].reference_pulse_shape = pixel_settings[
                    'refshape'].astype('float64')
                data.mc.tel[tel_id].meta['refstep'] = float(
                    pixel_settings['ref_step'])
                data.mc.tel[tel_id].time_slice = float(
                    pixel_settings['time_slice'])

                data.mc.tel[tel_id].photo_electron_image = array_event.get(
                    'photoelectrons', {}
                ).get(tel_id)
                if data.mc.tel[tel_id].photo_electron_image is None:
                    data.mc.tel[tel_id].photo_electron_image = np.zeros(
                        (n_pixel,), dtype='i2')

                tracking_position = tracking_positions[tel_id]
                data.mc.tel[tel_id].azimuth_raw = tracking_position[
                    'azimuth_raw']
                data.mc.tel[tel_id].altitude_raw = tracking_position[
                    'altitude_raw']
                data.mc.tel[tel_id].azimuth_cor = tracking_position.get(
                    'azimuth_cor', 0)
                data.mc.tel[tel_id].altitude_cor = tracking_position.get(
                    'altitude_cor', 0)
            yield data
            counter += 1
            if max_events and counter >= max_events:
                return
