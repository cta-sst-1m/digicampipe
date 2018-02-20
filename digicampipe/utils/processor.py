from abc import ABC, abstractmethod


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
