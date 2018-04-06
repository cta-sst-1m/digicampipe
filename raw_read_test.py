
from digicampipe.io.event_stream import event_stream
import astropy.units as u
from digicampipe.utils import geometry
from cts_core.camera import Camera


directory = '../../sst-1m_data/20171030/'
filename = directory + 'SST1M01_0_000.%03d.fits.fz'
file_list = [filename % number for number in range(19, 19 + 1)]
digicam_config_file = 'digicampipe/tests/resources/camera_config.cfg' 

# Source coordinates (in camera frame)
source_x = 0. * u.mm
source_y = 0. * u.mm

# Camera and Geometry objects (mapping, pixel, patch + x,y coordinates pixels)
digicam = Camera(_config_file=digicam_config_file)
digicam_geometry = geometry.generate_geometry_from_camera(camera=digicam, source_x=source_x, source_y=source_y)





data_stream = event_stream(file_list=file_list, expert_mode=False, camera_geometry=digicam_geometry, camera=digicam)

for event in data_stream:
    print(event)
