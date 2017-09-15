from digicampipe.calib.camera import filter, r1
from digicampipe.io.event_stream import event_stream
from digicamviewer.viewer import EventViewer
import numpy as np
import matplotlib.pyplot as plt


if __name__ == '__main__':

    directory = '/home/alispach/blackmonkey/calib_data/first_light/20170831/'
    filename = directory + 'CameraDigicam@sst1mserver_0_000.%d.fits.fz'
    file_list = [filename % number for number in range(10, 165)]
    data_stream = event_stream(file_list=file_list, expert_mode=True)
    unwanted_patch = [391, 392, 403, 404, 405, 416, 417]

    data_stream = filter.filter_patch(data_stream, unwanted_patch=unwanted_patch)
    data_stream = r1.calibrate_to_r1(event_stream=data_stream)

    n_events = 100000
    n_pixels = 1296
    baseline_mean = np.zeros((n_pixels, n_events))
    baseline_std = np.zeros((n_pixels, n_events))
    time = np.zeros(n_events)

    for i, event in zip(range(n_events), data_stream):

        for telescope_id in event.r0.tels_with_data:

            print(i)
            baseline_mean[..., i] = list(event.r1.tel[telescope_id].pedestal_mean.values())
            baseline_std[..., i] = list(event.r1.tel[telescope_id].pedestal_std.values())
            time[i] = event.r0.tel[telescope_id].local_camera_clock

    pixels = [0, 300, 1200]
    fig1 = plt.figure()
    fig2 = plt.figure()

    axis_1 = fig1.add_subplot(111)
    axis_2 = fig2.add_subplot(111)

    for pixel in pixels:

        axis_1.plot(time - time[0], baseline_mean[pixel], linestyle='-', label='pixel %d' % pixel)
        axis_2.plot(time - time[0], baseline_std[pixel], linestyle='-', label='pixel %d' % pixel)

    axis_1.set_xlabel('time [ns]')
    axis_2.set_xlabel('time [ns]')
    axis_1.set_ylabel('baseline [LSB]')
    axis_2.set_ylabel('std [LSB]')
    axis_1.legend()
    axis_2.legend()

    plt.figure()
    plt.hist(np.mean(baseline_mean, axis=-1), bins='auto')
    plt.xlabel('baseline [LSB]')

    plt.figure()
    plt.hist(np.mean(baseline_std, axis=-1), bins='auto')
    plt.xlabel('std [LSB]')

    plt.show()
    camera_config_file = '/home/alispach/Documents/PhD/ctasoft/CTS/config/camera_config.cfg'
    display = EventViewer(data_stream, camera_config_file=camera_config_file, scale='lin')
    display.draw()
    0/0