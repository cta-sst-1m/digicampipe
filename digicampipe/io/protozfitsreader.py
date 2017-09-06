#!/bin/env python
# this code should run in python3.
# Zfits/protobuf loader.
# import protozfitsreader
import numpy
import rawzfitsreader


class ZFile(object):
    def __init__(self, fname):
        # Save filename
        self.fname = fname
        # Read non-iterable tables (header), keep their contents in memory.
        # self.read_runheader()

    ### INTERNAL METHODS ###########################################################

    def _read_file(self):
        # Read file. Return a serialized string
        try:
            assert (self.ttype in ["RunHeader", "Events", "RunTails"])
        except AssertionError as e:
            print("Error: Table type not RunHeader, Events or RunTails")
            raise
        else:
            rawzfitsreader.open("%s:%s" % (self.fname, self.ttype))

    def _read_message(self):
        # Read next message. Fills property self.rawmessage and self.numrows
        self.rawmessage = rawzfitsreader.readEvent()
        self.numrows = rawzfitsreader.getNumRows()

    def _extract_field(self, obj, field):
        # Read a specific field in object 'obj' given as input 'field'
        if obj.HasField(field) == False:
            raise Exception("No field %s found in object %s" % (field, str(obj)))
        return (getattr(obj, field))

    def _get_numpyfield(self, field):
        try:
            numpyfield = toNumPyArray(field)
        except Exception as e:
            err = "Conversion to NumpyArray failed with error %s" % e
            raise Exception(err)
        else:
            return (numpyfield)

    ### PUBLIC METHODS #############################################################

    def list_tables(self):
        try:
            return (rawzfitsreader.listAllTables(self.fname))
        except:
            raise

    def read_runheader(self):
        # Get number of events in file
        import L0_pb2
        try:
            assert (self.ttype == "RunHeader")
        except:
            self.ttype = "RunHeader"
            self._read_file()

        self._read_message()
        self.header = L0_pb2.CameraRunHeader()
        self.header.ParseFromString(self.rawmessage)
        # self.print_listof_fields(self.header)

    def read_event(self):
        # read an event. Assume it is a camera event
        # C++ return a serialized string, python protobuf rebuild message from serial string
        import L0_pb2
        try:
            assert (self.ttype == "Events")
        except:
            self.ttype = "Events"
            self._read_file()
            self.eventnumber = 1
        else:
            self.eventnumber += 1

        # print("Reading event number %d" %self.eventnumber)
        self._read_message()
        self.event = L0_pb2.CameraEvent()
        self.event.ParseFromString(self.rawmessage)
        # self.print_listof_fields(self.event)

    def rewind_table(self):
        # Rewind the current reader. Go to the beginning of the table.
        rawzfitsreader.rewindTable()

    def move_to_next_event(self):
        # Iterate over events
        i = 0
        numrows = i + 2
        # Hook to deal with file with no header (1)
        if hasattr(self, 'numrows'):
            numrows = self.numrows
        # End - Hook to deal with file with no header (1)

        while i < numrows:
            self.read_event()
            # Hook to deal with file with no header (2)
            if hasattr(self, 'numrows'):
                numrows = self.numrows
            # End - Hook to deal with file with no header (2)

            # Hook to deal with file with no header (3)
            try:
                run_id = self.get_run_id()
                event_id = self.get_event_id()
            except:
                run_id = 0
                event_id = self.eventnumber
                # print('No header in this file, run_id set to 0 and event_id set to event_number')
            # Hook to deal with file with no header (3)

            yield run_id, event_id
            i += 1

    def get_telescope_id(self):
        return (self._get_numpyfield(self.event.telescopeID))

    def get_event_number(self):
        return (self._get_numpyfield(self.event.eventNumber))

    def get_run_id(self):
        return (self._get_numpyfield(self.header.runNumber))

    def get_central_event_gps_time(self):
        timeSec = self.event.trig.timeSec
        timeNanoSec = self.event.trig.timeNanoSec
        return (timeSec, timeNanoSec)

    def get_local_time(self):
        timeSec = self.event.local_time_sec
        timeNanoSec = self.event.local_time_nanosec
        return (timeSec, timeNanoSec)

    def get_event_number(self):
        return (self._get_numpyfield(self.event.arrayEvtNum))

    def get_event_type(self):
        return self.event.event_type

    def get_eventType(self):
        return self.event.eventType

    def get_num_channels(self):
        return (self._get_numpyfield(self.event.head.numGainChannels))

    def _get_adc(self, channel, telescope_id=None):
        # Expect hi/lo -> Will append Gain at the end -> hiGain/loGain
        sel_channel = self._extract_field(self.event, "%sGain" % channel)
        return (sel_channel)

    def get_pixel_position(self, telescope_id=None):
        # TODO: Not implemented yet
        return (None)

    def get_number_of_pixels(self, telescope_id=None):
        # TODO: Not implemented yet
        return (None)

    def get_adc_sum(self, channel, telescope_id=None):
        sel_channel = self._get_adc(channel, telescope_id)
        integrals = sel_channel.integrals

        nsamples = self._get_numpyfield(integrals.num_samples)
        pixelsIndices = self._get_numpyfield(integrals.pixelsIndices)

        # Structured array (dict of dicts)
        properties = dict()
        for par in ["gains", "maximumTimes", "raiseTimes", "tailTimes", "firstSplIdx"]:
            properties[par] = dict(zip(pixelIndices, self._get_numpyfield(_extract_field(par))))
        return (properties)

    def get_adc_samples(self, channel, telescope_id=None):
        sel_channel = self._get_adc(channel, telescope_id)
        waveforms = sel_channel.waveforms
        samples = self._get_numpyfield(waveforms.samples)
        pixels = self._get_numpyfield(waveforms.pixelsIndices)
        npixels = len(pixels)
        # Structured array (dict)
        samples = samples.reshape(npixels, -1)
        properties = dict(zip(pixels, samples))

        return (properties)

    def get_adcs_samples(self, telescope_id=None):
        '''
        Get the samples for all channels

        :param telescope_id: id of the telescopeof interest
        :return: dictionnary of samples (value) per pixel indices (key)
        '''
        waveforms = self.event.hiGain.waveforms
        samples = self._get_numpyfield(waveforms.samples)
        pixels = self._get_numpyfield(waveforms.pixelsIndices)
        npixels = len(pixels)
        # Structured array (dict)
        samples = samples.reshape(npixels, -1)
        properties = dict(zip(pixels, samples))
        return (properties)

    def get_trigger_input_traces(self, telescope_id=None):
        '''
        Get the samples for all channels

        :param telescope_id: id of the telescopeof interest
        :return: dictionnary of samples (value) per pixel indices (key)
        '''

        patch_traces = self._get_numpyfield(self.event.trigger_input_traces)
        patches = numpy.arange(0, 192, 1)  # TODO check : might not be correct yet
        patch_traces = patch_traces.reshape(patches.shape[0], -1)
        properties = dict(zip(patches, patch_traces))

        return (properties)

    def get_trigger_output_patch7(self, telescope_id=None):
        '''
        Get the samples for all channels

        :param telescope_id: id of the telescopeof interest
        :return: dictionnary of samples (value) per pixel indices (key)
        '''
        frames = self._get_numpyfield(self.event.trigger_output_patch7)
        n_samples = frames.shape[0] / 18 / 3
        frames = numpy.unpackbits(frames.reshape(n_samples, 3, 18, 1), axis=-1)[..., ::-1].reshape(n_samples, 3,
                                                                                                   144).reshape(
            n_samples, 432).T
        patches = numpy.arange(0, 432)  # TODO access patch_ids from self.event
        properties = dict(zip(patches, frames))
        return (properties)

    def get_trigger_output_patch19(self, telescope_id=None):
        '''
        Get the samples for all channels

        :param telescope_id: id of the telescopeof interest
        :return: dictionnary of samples (value) per pixel indices (key)
        '''
        frames = self._get_numpyfield(self.event.trigger_output_patch19)
        n_samples = frames.shape[0] / 18 / 3
        frames = numpy.unpackbits(frames.reshape(n_samples, 3, 18, 1), axis=-1)[..., ::-1].reshape(n_samples, 3,
                                                                                                   144).reshape(
            n_samples, 432).T
        patches = numpy.arange(0, 432)  # TODO acess patch_ids from self.event
        properties = dict(zip(patches, frames))

        return (properties)

    def get_pixel_flags(self, telescope_id=None):
        '''
        Get the flag of pixels
        :param id of the telescopeof interest
        :return: dictionnary of flags (value) per pixel indices (key)
        '''
        waveforms = self.event.hiGain.waveforms
        flags = self._get_numpyfield(self.event.pixels_flags)
        pixels = self._get_numpyfield(waveforms.pixelsIndices)
        npixels = len(pixels)
        # Structured array (dict)
        properties = dict(zip(pixels, flags))
        return (properties)

    def print_listof_fields(self, obj):
        fields = [f.name for f in obj.DESCRIPTOR.fields]
        print(fields)
        return (fields)


# below are utility functions used to convert from AnyArray to numPyArrays
def typeNone(data):
    raise Exception("This any array has no defined type")


def typeS8(data):
    return numpy.fromstring(data, numpy.int8)


def typeU8(data):
    return numpy.fromstring(data, numpy.uint8)


def typeS16(data):
    return numpy.fromstring(data, numpy.int16)


def typeU16(data):
    return numpy.fromstring(data, numpy.uint16)


def typeS32(data):
    return numpy.fromstring(data, numpy.int32)


def typeU32(data):
    return numpy.fromstring(data, numpy.uint32)


def typeS64(data):
    return numpy.fromstring(data, numpy.int64)


def typeU64(data):
    return numpy.fromstring(data, numpy.uint64)


def typeFloat(data):
    return numpy.fromstring(data, numpy.float)


def typeDouble(data):
    return numpy.fromstring(data, numpy.double)


def typeBool(any_array):
    raise Exception("I have no idea if the boolean representation of the anyarray is the same as the numpy one")


artificialSwitchCase = {0: typeNone,
                        1: typeS8,
                        2: typeU8,
                        3: typeS16,
                        4: typeU16,
                        5: typeS32,
                        6: typeU32,
                        7: typeS64,
                        8: typeU64,
                        9: typeFloat,
                        10: typeDouble,
                        11: typeBool,
                        }


def toNumPyArray(any_array):
    return artificialSwitchCase[any_array.type](any_array.data)
