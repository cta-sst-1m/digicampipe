from digicampipe.io.hdf5 import digicamtoy_event_source
import pkg_resources
import os

example_file_paths = []

for i in range(12):

    path = pkg_resources.resource_filename(
        'digicampipe',
        os.path.join(
            'tests',
            'resources',
            'digicamtoy',
            'test_digicamtoy_{}.hdf5'.format(i)
        )
    )

    example_file_paths.append(path)

def test_event_source_new_style():

    for _ in digicamtoy_event_source(example_file_paths[0]):
        pass
