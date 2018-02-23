from digicampipe.io.event_stream import event_stream
from abc import ABC, abstractmethod

# from . import filters
# from . import calib
# from . import baseline
# from . import io


class Processor(ABC):
    ''' just a callable class which:
        - takes an "event"
        - performs some "analysis"
        - attaches the result back to the event.
    '''

    @abstractmethod
    def __call__(self, event):
        # calculate something and **attach** it to event
        return event


class SkipEvent(Exception):
    '''Cause Process to skip this event.

    Processors raise this Exception in their __call__() in case they
    want to signal the process to skip this entire event.

    This exception is similar to the build-in StopIteration, just that
    StopIteration causes a for-loop to end, while this SkipEvent causes
    the for-loop to `continue` with the next event.
    '''
    pass


def run_process(process, files):
    for event in event_stream(files):
        try:
            for processor in process:
                event = processor(event)
        except SkipEvent:
                continue
