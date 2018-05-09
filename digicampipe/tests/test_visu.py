import numpy as np
from digicampipe.visualization.plot import plot_array_camera


def test_array_to_camera_view():

    image = np.random.normal(size=1296)
    plot_array_camera(image, 'some random numbers', limits=(-1, 1))