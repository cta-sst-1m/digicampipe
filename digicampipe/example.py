from digicampipe.io.event_stream import event_stream
from digicamviewer import EventViewer


def from_level_x_to_level_y(stream):  # e.g. from r0 to r1

    for event in stream:  # loop over the events

        something = compute_something(event)  # e.g. compute baseline
        add_container_of_level_y(event)  # e.g. add r1 container
        event.level_y = apply(event, something)  # e.g. substract baseline

        yield event  # e.g. return an iterator


if __name__ == '__main__':

    file_list = ['first_light_1.zfits.fz', 'first_light_2.zfits.fz']  # my list of files
    event_stream = event_stream(url=file_list)  # a reader for my files
    event_stream = from_level_x_to_level_y(event_stream=event_stream)  # apply step y to level x

    view = EventViewer(event_stream)
    view.draw()  # view the results











