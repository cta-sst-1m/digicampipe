from digicampipe.io import zfits


def event_stream(file_list, expert_mode=False):

    for file in file_list:

        event_stream = zfits.zfits_event_source(url=file, expert_mode=expert_mode)

        for event in event_stream:

            yield event

