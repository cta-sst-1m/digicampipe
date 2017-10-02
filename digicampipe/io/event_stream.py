from digicampipe.io import zfits


def event_stream(file_list, expert_mode=False, geom_file=None, remapped=False):

    i = 0

    for file in file_list:

        event_stream = zfits.zfits_event_source(url=file, expert_mode=expert_mode, geom_file=geom_file, remapped=remapped)

        print(file)
        for event in event_stream:

            yield event

