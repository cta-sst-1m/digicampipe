from digicampipe.io import event_stream
from cts_core.camera import Camera
from digicampipe.utils import geometry
from digicamviewer.viewer import EventViewer2
import matplotlib.pyplot as plt


if __name__ == '__main__':

    # directory = '/home/alispach/data/BASELINE_TEST/'
    directory = '/home/alispach/data/digicam_commissioning/trigger/mc/'
    # filename = directory + 'BASELINE_TEST_0_000.00%d.fits.fz'
    filename = directory + 'nsb_full_camera_%d.hdf5'
    file_list = [filename % number for number in range(15, 16)]
    digicam_config_file = '/home/alispach/ctasoft/CTS/config/camera_config.cfg'
    max_events = 100

    digicam = Camera(_config_file=digicam_config_file)
    digicam_geometry = geometry.generate_geometry_from_camera(camera=digicam)

    data_stream = event_stream.event_stream(file_list=file_list,
                                            camera_geometry=digicam_geometry,
                                            max_events=max_events,
                                            camera=digicam,
                                            expert_mode=True)

    with plt.style.context('ggplot'):
        display = EventViewer2(data_stream, n_samples=51, camera_config_file=digicam_config_file, scale='lin')
        display.draw()
        plt.show()

    for data in data_stream:

        pass