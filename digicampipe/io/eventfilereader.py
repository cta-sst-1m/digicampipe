from ctapipe.io.eventfilereader import EventFileReader
from digicampipe.io.zfits import zfits_event_source, zfits_get_list_event_ids


class ZfitsFileReader(EventFileReader):
    name = 'ZfitsFileReader'
    origin = 'zfits'


    @staticmethod
    def check_file_compatibility(file_path):
        compatible = True
        # TODO: Change check to be a try of hessio_event_source?
        if not file_path.endswith('.fz'):
            compatible = False
        return compatible

    @property
    def num_events(self):
        self.log.info("Obtaining number of events in file...")
        if self._num_events:
            pass
        else:
            self._num_events = len(self.event_id_list)
        self.log.info("Number of events inside file = {}"
                      .format(self._num_events))
        return self._num_events

    @property
    def event_id_list(self):
        self.log.info("Retrieving list of event ids...")
        if self._event_id_list:
            pass
        else:
            self.log.info("Building new list of event ids...")
            ids = zfits_get_list_event_ids(self.input_path,
                                            max_events=self.max_events)
            self._event_id_list = ids
        self.log.info("List of event ids retrieved.")
        return self._event_id_list

    def read(self, allowed_tels=None, expert_mode=True, requested_event=None,
             use_event_id=False):
        """
        Read the file using the appropriate method depending on the file origin
        Parameters
        ----------
        allowed_tels : list[int]
            select only a subset of telescope, if None, all are read. This can
            be used for example emulate the final CTA data format, where there
            would be 1 telescope per file (whereas in current monte-carlo,
            they are all interleaved into one file)
        requested_event : int
            Seek to a paricular event index
        use_event_id : bool
            If True ,'requested_event' now seeks for a particular event id
            instead of index
        Returns
        -------
        source : generator
            A generator that can be iterated over to obtain events
        """

        # Obtain relevent source
        self.log.debug("Reading file...")
        if self.max_events:
            self.log.info("Max events being read = {}".format(self.max_events))
        source = zfits_event_source(self.input_path,
                                     max_events=self.max_events,
                                     allowed_tels=allowed_tels,
                                    expert_mode=expert_mode)

        self.log.debug("File reading complete")
        return source