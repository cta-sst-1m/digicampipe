# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Components to read ZFITS data.
This requires the protozfitsreader python library to be installed
"""
import logging
from digicampipe.io.containers import DataContainer
from digicampipe.io.containers import DigiCamCameraContainer, DigiCamExpertCameraContainer
from digicampipe.utils import geometry

logger = logging.getLogger(__name__)

try:
    import digicampipe.io.protozfitsreader as protozfitsreader
except ImportError as err:
    logger.fatal("the `protozfitsreader` python module is required to access MC data: {}"
                 .format(err))
    raise err

__all__ = [
    'zfits_event_source',
]

def zfits_event_source(url, max_events=None, allowed_tels=None, expert_mode = False, geom_file = None):
    """A generator that streams data from an ZFITs data file
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
    """

    # load the zfits file
    try:
        zfits = protozfitsreader.ZFile(url)
    except:
        raise RuntimeError("zfits_event_source failed to open '{}'"
                           .format(url))

    # intialise counter and event generator


    counter = 0
    eventstream = zfits.move_to_next_event()

    # print('File %s contains %d events' % (url, zfits.numrows))


    #print(eventstream)
    # loop over the events
    for run_id, eventid in eventstream:

        #print('hello')
        # define the main container and fill some metadata
        data = DataContainer()
        data.meta['zfits__input'] = url
        data.meta['zfits__max_events'] = max_events
        data.r0.run_id = run_id
        data.r0.event_id = zfits.get_event_number()
        data.r0.tels_with_data = [zfits.event.telescopeID, ]
        data.count = counter

        # remove forbidden telescopes
        if allowed_tels:
            data.r0.tels_with_data = \
                [list(filter(lambda x: x in data.r0.tels_with_data, sublist)) for sublist in allowed_tels]

        for tel_id in data.r0.tels_with_data :
            # TODO: add the time flag
            data.inst.num_channels[tel_id] = zfits.event.num_gains
            data.inst.num_pixels[tel_id] = zfits._get_numpyfield(zfits.event.hiGain.waveforms.pixelsIndices).shape[0]
            data.inst.pixel_pos[tel_id] = geometry.find_pixel_positions(geom_file)

            if data.inst.num_pixels[tel_id] == 1296:
                # Note, I'll add in the data model of the zfits a camera identifier, just need some time
                # to be released and to have some data containing this new field to test.
                # In the future telescopeID will allow to know which camera it is
                data.r0.tel[tel_id] = DigiCamCameraContainer() if not expert_mode else DigiCamExpertCameraContainer()

                data.r0.tel[tel_id].camera_event_number = zfits.event.eventNumber
                #data.r0.tel[tel_id].pixel_flags = zfits.get_pixel_flags(telescope_id=tel_id)

                seconds, nano_seconds = zfits.get_local_time()
                data.r0.tel[tel_id].local_camera_clock = (seconds * 1e9 + nano_seconds * 4)
                seconds, nano_seconds = zfits.get_central_event_gps_time()
                data.r0.tel[tel_id].gps_time = (seconds * 1e9 + nano_seconds * 4)
                #print(zfits.get_event_type(),zfits.get_eventType())
                data.r0.tel[tel_id].event_type =zfits.get_event_type()
                data.r0.tel[tel_id].eventType =zfits.get_eventType()

                if expert_mode:
                    data.r0.tel[tel_id].trigger_input_traces = zfits.get_trigger_input_traces(telescope_id=tel_id)
                    data.r0.tel[tel_id].trigger_output_patch7 = zfits.get_trigger_output_patch7(telescope_id=tel_id)
                    data.r0.tel[tel_id].trigger_output_patch19 = zfits.get_trigger_output_patch19(telescope_id=tel_id)


            # elif data.inst.num_pixels ==  N :
            #    data.dl0.tel[tel_id] = OtherCameraNameRawCameraContainer()
            # if no specific OtherCameraNameRawCameraContainer() it satys by default a RawCameraContainer

            data.r0.tel[tel_id].num_samples = zfits._get_numpyfield(zfits.event.hiGain.waveforms.samples).shape[0] //\
                                               zfits._get_numpyfield(zfits.event.hiGain.waveforms.pixelsIndices).shape[0]
            data.r0.tel[tel_id].adc_samples = zfits.get_adcs_samples(telescope_id=tel_id)

            #print(data.r0.tel[tel_id].adc_samples)
            #nchans = zfits.get_num_channels(tel_id)
            #data.inst.num_channels[tel_id] = nchans

        yield data
        counter += 1

    if max_events is not None and counter > max_events:
        return