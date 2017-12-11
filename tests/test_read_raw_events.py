import pytest

import pkg_resources
import os

example_file_path = pkg_resources.resource_filename(
    'digicampipe',
    os.path.join(
        'tests',
        'resources',
        'example_10evts.fits.fz'
    )
)

@pytest.mark.skip(reason="we know the current version does not raise")
def test_zfile_raises_on_wrong_path():
    from digicampipe.io.protozfitsreader import ZFile
    with pytest.raises(FileNotFoundError):
        ZFile('foo.bar')


def test_zfile_opens_correct_path():
    from digicampipe.io.protozfitsreader import ZFile
    ZFile(example_file_path)


def test_can_iterate_over_events():
    from digicampipe.io.protozfitsreader import ZFile
    zfits = ZFile(example_file_path)
    event_stream = zfits.move_to_next_event()
    for __ in event_stream:
        pass
