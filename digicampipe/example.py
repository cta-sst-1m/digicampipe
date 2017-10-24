
from digicampipe.io.event_stream import event_stream
from digicampipe.utils.geometry import generate_geometry_from_camera
from digicampipe.calib.camera import filter
from digicampipe.io.save_adc import save_dark
from digicamviewer.viewer import EventViewer2
from cts_core.camera import Camera
import astropy.units as u


if __name__ == '__main__':

    directory = '/my_path_to_data/'
    filename = directory + 'CRAB_01_0_000.%03d.fits.fz'
    file_list = [filename % number for number in [2]]
    camera_config_file = 'path_to_config_file/camera_config.cfg'
    dark_filename = 'dark.npz'

    unwanted_patch = [306, 318, 330, 342, 200]

    digicam = Camera(_config_file=camera_config_file)
    digicam_geometry = generate_geometry_from_camera(camera=digicam)

    data_stream = event_stream(file_list=file_list, camera_geometry=digicam_geometry)
    data_stream = filter.set_patches_to_zero(data_stream, unwanted_patch=unwanted_patch)
    data_stream = filter.filter_event_types(data_stream, flags=[8])
    data_stream = filter.filter_period(event_stream, period=10*u.second)
    data_stream = save_dark(data_stream, directory + dark_filename)

    view = EventViewer2(event_stream)
    view.draw()








