import numpy as np
import astropy.units as u


def filter_patch(event_stream, unwanted_patch):

    for event in event_stream:

        for telescope_id in event.r0.tels_with_data:

            r0_camera = event.r0.tel[telescope_id]

            output_trigger_patch7 = np.array(list(r0_camera.trigger_output_patch7.values()))

            patch_condition = np.any(output_trigger_patch7[unwanted_patch])

            if not patch_condition:
                # Set the event type
                event.trig.trigger_flag = 0
            else:
                # Set the event type
                event.trig.trigger_flag = 1

        yield event


def set_pixels_to_zero(event_stream, unwanted_pixels):

    for event in event_stream:

        for telescope_id in event.r0.tels_with_data:

            r0_camera = event.r0.tel[telescope_id]
            adc_samples = np.array(list(r0_camera.adc_samples.values()))
            adc_samples[unwanted_pixels] = 0

            r0_camera.adc_samples = dict(zip(range(adc_samples.shape[0]), adc_samples))

            yield event


def fill_flag(event_stream, unwanted_patch=None):

    for event in event_stream:

        for telescope_id in event.r0.tels_with_data:

            r0_camera = event.r0.tel[telescope_id]

            if unwanted_patch is not None:

                output_trigger_patch7 = np.array(list(r0_camera.trigger_output_patch7.values()))

                patch_condition = np.any(output_trigger_patch7[unwanted_patch])

                if not patch_condition:
                    # Set the event type
                    r0_camera.event_type_1 = 8
                else:
                    # Set the event type
                    r0_camera.event_type_1 = 0

        yield event


def filter_event_types(event_stream, flags=[0]):

    for event in event_stream:

        for telescope_id in event.r0.tels_with_data:
            flag = event.r0.tel[telescope_id].event_type_1

            if flag in flags:

                yield event


def filter_shower(event_stream, min_photon):
    """
    Filter events as a function of the processing level
    :param event_stream:
    :param min_photon:
    :return:
    """
    for event in event_stream:

        for telescope_id in event.r0.tels_with_data:
            dl1_camera = event.dl1.tel[telescope_id]
            if np.sum(dl1_camera.pe_samples[dl1_camera.cleaning_mask]) >= min_photon:
                yield event


def filter_shower_adc(event_stream, min_adc):
    """
    Filter events as a function of the processing level
    :param event_stream:
    :param min_adc:
    :return:
    """
    for event in event_stream:

        for telescope_id in event.r0.tels_with_data:
            r1_camera = event.r1.tel[telescope_id]
            if np.sum(np.max(r1_camera.adc_samples, axis=-1)) >= min_adc:
                yield event


def filter_missing_baseline(event_stream):

    for event in event_stream:

        for telescope_id in event.r0.tels_with_data:

            r0_camera = event.r0.tel[telescope_id]

            condition = np.all(np.isnan(r0_camera.baseline))

            if not condition:

                yield event


def filter_trigger_time(event_stream, time):

    for event in event_stream:

        for telescope_id in event.r0.tels_with_data:

            r0_camera = event.r0.tel[telescope_id]

            output_trigger_patch7 = np.array(list(r0_camera.trigger_output_patch7.values()))

            condition = np.sum(output_trigger_patch7) > time

            if condition:

                yield event

def filter_period(event_stream, period):

    t_last = 0 * u.second

    for event in event_stream:

        for telescope_id in event.r0.tels_with_data:

            t_new = event.r0.tel[telescope_id].local_camera_clock * u.nanosecond

            if (t_new - t_last) > period:

                t_last = t_new
                yield event

from gzip import open as gzip_open
import pickle

def load_from_pickle_gz(file):
    file=gzip_open( file, "rb" )
    while True:
        try:
            yield pickle.load(file)
        except (EOFError,pickle.UnpicklingError):
            return

from os.path import isfile
from ctapipe.io.serializer import Serializer

from os import remove

def save_to_pickle_gz(event_stream, file, overwrite=False, max_events=None):
    if isfile(file):
        if overwrite:
            print('remove old', file, 'file')
            remove(file)
        else:
            print(file, 'exist, exiting...')
            return
    writer = Serializer(filename=file, format='pickle', mode='w')
    counter_events=0
    for event in event_stream:
        writer.add_container(event)
        counter_events+=1
        if max_events!=None and counter_events>=max_events:
            break
    writer.close()
