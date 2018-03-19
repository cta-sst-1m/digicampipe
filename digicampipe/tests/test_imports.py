# This just checks some imports work


def test_imports():

    from digicampipe.reco.camera import filter, r1, random_triggers, dl0, dl2, dl1
    from digicampipe.io.event_stream import event_stream
    from digicampipe.utils import geometry
    from cts_core.camera import Camera
    from digicampipe.io.save_hillas import save_hillas_parameters
    from digicampipe.io.save_hillas import save_hillas_parameters
    from digicampipe.visualization import EventViewer
    from digicampipe.utils import utils
    import protozfitsreader
    from protozfitsreader import rawzfitsreader
    from protozfitsreader import ZFile
    from histogram.histogram import Histogram1D
