# This just checks some imports work


def test_imports():

    # TODO make real unit tests for each of these imports

    from digicampipe.calib.camera import (
        filter, r1, random_triggers, dl2, dl1)
    from digicampipe.io.event_stream import event_stream
    from digicampipe.utils import geometry
    from cts_core.camera import Camera
    from digicampipe.io.save_hillas import save_hillas_parameters
    from digicampipe.io.save_hillas import save_hillas_parameters
    from digicampipe.visualization import EventViewer
    from digicampipe.utils import utils
