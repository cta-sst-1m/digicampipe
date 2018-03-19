
def calibrate_to_dl0(event_stream):

    for i, event in enumerate(event_stream):

        for telescope_id in event.r0.tels_with_data:

            pass

        yield event
