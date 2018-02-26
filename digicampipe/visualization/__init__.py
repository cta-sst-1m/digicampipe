import numpy as np
import sys
from digicampipe.visualization import mpl as visualization
from . import geometry
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter, MaxNLocator
from matplotlib.widgets import Button, RadioButtons, CheckButtons
from itertools import cycle
from matplotlib.colors import LogNorm
import matplotlib as mpl
from cts_core import camera


class EventViewer():

    def __init__(self, event_stream, camera_config_file, n_samples, scale='lin', limits_colormap=None, limits_readout=None, time_bin_start=0, pixel_id_start=0):

        mpl.figure.autolayout = False
        self.first_call = True
        self.event_stream = event_stream
        self.scale = scale
        self.limits_colormap = limits_colormap if limits_colormap is not None else [-np.inf, np.inf]
        self.limits_readout = limits_readout
        self.time_bin = time_bin_start
        self.pixel_id = pixel_id_start
        self.mask_pixels = False
        self.hillas = False

        self.event_clicked_on = EventClicked(pixel_start=self.pixel_id)
        self.camera = camera.Camera(_config_file=camera_config_file)
        self.geometry = geometry.generate_geometry(camera=self.camera)[0]
        self.n_pixels = len(self.camera.Pixels)
        self.n_samples = n_samples
        self.cluster_matrix = np.zeros((len(self.camera.Clusters_7), len(self.camera.Clusters_7)))

        for cluster in self.camera.Clusters_7:

            for patch in cluster.patchesID:
                self.cluster_matrix[cluster.ID, patch] = 1

        self.event_id = None
        self.r0_container = None
        self.r1_container = None
        self.dl0_container = None
        self.dl1_container = None
        self.dl2_container = None
        self.trigger_output = None
        self.trigger_input = None
        self.trigger_patch = None
        self.nsb = [np.nan]*self.n_pixels
        self.gain_drop = [np.nan]*self.n_pixels
        self.baseline = [np.nan]*self.n_pixels
        self.std = [np.nan]*self.n_pixels
        self.flag = None

        self.readout_view_types = ['raw', 'baseline substracted', 'photon', 'trigger input', 'trigger output', 'cluster 7', 'reconstructed charge']
        self.readout_view_type = 'raw'

        self.camera_view_types = ['sum', 'std', 'mean', 'max', 'time']
        self.camera_view_type = 'std'

        self.figure = plt.figure(figsize=(20, 10))
        self.axis_readout = self.figure.add_subplot(122)
        self.axis_camera = self.figure.add_subplot(121)
        self.axis_camera.axis('off')

        self.axis_readout.set_xlabel('t [ns]')
        self.axis_readout.set_ylabel('[ADC]')
        self.axis_readout.legend(loc='upper right')
        self.axis_readout.yaxis.set_major_formatter(FormatStrFormatter('%d'))
        self.axis_readout.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=10))

        self.trace_readout, = self.axis_readout.step(np.arange(self.n_samples) * 4, np.ones(self.n_samples), where='mid')
        self.trace_time_plot, = self.axis_readout.plot(np.array([self.time_bin, self.time_bin]) * 4, np.ones(2), color='k',
                                                       linestyle='--')

        self.camera_visu = visualization.CameraDisplay(self.geometry, ax=self.axis_camera, title='', norm=self.scale,
                                                       cmap='viridis',
                                                       allow_pick=True)

        #if limits_colormap is not None:
        #    self.camera_visu.set_limits_minmax(limits_colormap[0], limits_colormap[1])

        self.camera_visu.image = np.zeros(self.n_pixels)
        self.camera_visu.cmap.set_bad(color='k')
        self.camera_visu.add_colorbar(orientation='horizontal', pad=0.03, fraction=0.05, shrink=.85)

        #if self.scale == 'log':
        #    self.camera_visu.colorbar.set_norm(LogNorm(vmin=1, vmax=None, clip=False))
        self.camera_visu.colorbar.set_label('[LSB]')
        self.camera_visu.axes.get_xaxis().set_visible(False)
        self.camera_visu.axes.get_yaxis().set_visible(False)
        self.camera_visu.on_pixel_clicked = self.draw_readout
        self.camera_visu.pixels.set_snap(False)  # snap cursor to pixel center

#        self.camera_visu.pixels.set_picker(False)

        # Buttons

        self.axis_next_event_button = self.figure.add_axes([0.35, 0.9, 0.15, 0.07], zorder=np.inf)
        self.axis_next_camera_view_button = self.figure.add_axes([0., 0.85, 0.1, 0.15], zorder=np.inf)
        self.axis_next_view_type_button = self.figure.add_axes([0., 0.18, 0.1, 0.15], zorder=np.inf)
        self.axis_check_button = self.figure.add_axes([0.35, 0.18, 0.1, 0.1], zorder=np.inf)
        self.axis_next_camera_view_button.axis('off')
        self.axis_next_view_type_button.axis('off')
        self.button_next_event = Button(self.axis_next_event_button, 'Next')
        self.radio_button_next_camera_view = RadioButtons(self.axis_next_camera_view_button, self.camera_view_types,
                                                          active=self.camera_view_types.index(self.camera_view_type))
        self.radio_button_next_view_type = RadioButtons(self.axis_next_view_type_button, self.readout_view_types,
                                                        active=self.readout_view_types.index(self.readout_view_type))

        self.check_button = CheckButtons(self.axis_check_button, ('mask', 'hillas'), (self.mask_pixels, self.hillas))
        self.radio_button_next_view_type.set_active(self.readout_view_types.index(self.readout_view_type))
        self.radio_button_next_camera_view.set_active(self.camera_view_types.index(self.camera_view_type))

    def next(self, event=None, step=1):

        for i, event in zip(range(step), self.event_stream):
            pass

        telescope_id = event.r0.tels_with_data[0]
        self.event_id = event.r0.event_id
        self.r0_container = event.r0.tel[telescope_id]
        self.r1_container = event.r1.tel[telescope_id]
        self.dl0_container = event.dl0.tel[telescope_id]
        self.dl1_container = event.dl1.tel[telescope_id]
        self.dl2_container = event.dl2
        self.adc_samples = self.r0_container.adc_samples
        self.trigger_output = self.r0_container.trigger_output_patch7
        self.trigger_input = self.r0_container.trigger_input_traces

        try:

            self.baseline = self.r0_container.baseline if np.isnan(self.r0_container.digicam_baseline).all() else self.r0_container.digicam_baseline
            zero_image = np.zeros((self.n_pixels, self.n_samples))

            self.std = self.r0_container.standard_deviation if self.r0_container.standard_deviation is not None else np.nan * zero_image
            self.flag = self.r0_container.camera_event_type if self.r0_container.camera_event_type is not None else np.nan
            self.nsb = self.r1_container.nsb if self.r1_container.nsb is not None else np.nan * zero_image
            self.gain_drop = self.r1_container.gain_drop if self.r1_container.gain_drop is not None else np.nan * zero_image

        except:

            pass

        if self.first_call:

            self.first_call = False

        self.update()

    def update(self):

        self.draw_readout(self.pixel_id)
        self.draw_camera()
        self.button_next_event.label.set_text('Next : current event #%d' % (self.event_id))

    def draw(self):

        self.next()
        self.button_next_event.on_clicked(self.next)
        self.radio_button_next_camera_view.on_clicked(self.next_camera_view)
        self.radio_button_next_view_type.on_clicked(self.next_view_type)
        self.check_button.on_clicked(self.draw_on_camera)
        self.figure.canvas.mpl_connect('key_press_event', self.press)
        self.camera_visu._on_pick(self.event_clicked_on)
        plt.show()

    def draw_camera(self, plot_hillas=False):

        self.camera_visu.image = self.compute_image()

        if plot_hillas:
            self.camera_visu.overlay_moments(self.dl2_container.shower)

    def draw_readout(self, pixel):

        y = self.compute_trace()[pixel]
        limits_y = self.limits_readout if self.limits_readout is not None else [np.min(y), np.max(y) + 10]
        self.pixel_id = pixel
        self.event_clicked_on.ind[-1] = self.pixel_id
        self.trace_readout.set_ydata(y)

        legend = ''

        try:

            legend += ' flag = {},'.format(self.flag)

        except:

            pass

        legend += ' pixel = {},'.format(self.pixel_id)
        legend += ' bin = {} \n'.format(self.time_bin)

        try:

            legend += ' B = {:0.2f} [LSB],'.format(self.baseline[self.pixel_id])

        except:

            pass

        try:

            legend += ' $\sigma = $ {:0.2f} [LSB] \n'.format(self.std[self.pixel_id])

        except:

            pass

        try:

            legend += ' $G_{{drop}} = $ {:0.2f},'.format(self.gain_drop[self.pixel_id])
            legend += ' $f_{{nsb}} = $ {:0.2f} [GHz]'.format(self.nsb[self.pixel_id])

        except:

            pass

        self.trace_readout.set_label(legend)
        self.trace_time_plot.set_ydata(limits_y)
        self.trace_time_plot.set_xdata(self.time_bin * 4)
        self.axis_readout.set_ylim(limits_y)
        self.axis_readout.legend(loc='upper right')

        if self.readout_view_type in ['photon', 'reconstructed charge']:

            self.axis_readout.set_ylabel('[p.e.]')

        else:

            self.axis_readout.set_ylabel('[LSB]')

    def compute_trace(self):

        if self.readout_view_type in self.readout_view_types:

            if self.readout_view_type == 'raw':

                image = self.adc_samples

            elif self.readout_view_type == 'trigger output' and self.trigger_output is not None:

                image = np.array([self.trigger_output[pixel.patch] for pixel in self.camera.Pixels])

            elif self.readout_view_type == 'trigger input' and self.trigger_input is not None:

                image = np.array([self.trigger_input[pixel.patch] for pixel in self.camera.Pixels]) #np.zeros((self.n_pixels, self.n_samples))

            elif self.readout_view_type == 'cluster 7' and self.trigger_input is not None:

                trigger_input_patch = np.dot(self.cluster_matrix, self.trigger_input)
                image = np.array([trigger_input_patch[pixel.patch] for pixel in self.camera.Pixels])

            elif self.readout_view_type == 'photon' and self.dl1_container.pe_samples_trace is not None:

                image = self.dl1_container.pe_samples_trace

            elif self.readout_view_type == 'baseline substracted' and self.r1_container.adc_samples is not None:

                image = self.adc_samples - self.baseline[:, np.newaxis]

            elif self.readout_view_type == 'reconstructed charge' and (self.dl1_container.time_bin is not None or self.dl1_container.pe_samples is not None):

                image = np.zeros((self.n_pixels, self.n_samples))
                time_bins = self.dl1_container.time_bin
                image[time_bins] = self.dl1_container.pe_samples

            else:

                image = np.zeros((self.n_pixels, self.n_samples))

        return image

    def next_camera_view(self, camera_view, event=None):

        self.camera_view_type = camera_view
        if self.readout_view_type in ['photon','reconstructed charge']:
            self.camera_visu.colorbar.set_label('[p.e.]')

        else:
            self.camera_visu.colorbar.set_label('[LSB]')

        self.update()

    def next_view_type(self, view_type, event=None):

        self.readout_view_type = view_type

        if view_type in ['photon', 'reconstructed charge']:
            self.camera_visu.colorbar.set_label('[p.e.]')

        else:
            self.camera_visu.colorbar.set_label('[LSB]')

        self.update()

    def draw_on_camera(self, to_draw_on, event=None):

        if to_draw_on == 'hillas':

            if self.hillas:

                self.hillas = False

            else:

                self.hillas = True

        if to_draw_on == 'mask':

            if self.mask_pixels:

                self.mask_pixels = False

            else:

                self.mask_pixels = True

        self.update()

    def set_time(self, time):

        if time < self.n_samples and time >= 0:

            self.time_bin = time
            self.update()

    def set_pixel(self, pixel_id):

        if pixel_id < self.n_samples and pixel_id >= 0:
            self.pixel_id = pixel_id
            self.update()

    def compute_image(self):

        image = self.compute_trace()

        if self.camera_view_type in self.camera_view_types:

            if self.camera_view_type == 'mean':

                self.image = np.mean(image, axis=1)

            elif self.camera_view_type == 'std':

                self.image = np.std(image, axis=1)

            elif self.camera_view_type == 'max':

                self.image = np.max(image, axis=1)

            elif self.camera_view_type == 'sum':

                self.image = np.sum(image, axis=1)

            elif self.camera_view_type == 'time':

                self.image = image[:, self.time_bin]

        else:

            print('Cannot compute for camera type : %s' % self.camera_view)

        if self.limits_colormap is not None:

            mask = (self.image >= self.limits_colormap[0])
            if not self.limits_colormap[1] == np.inf:
                image[(self.image > self.limits_colormap[1])] = self.limits_colormap[1]

        if self.mask_pixels:

            #mask = mask * self.dl1_container.cleaning_mask
            mask = mask * self.dl1_container.cleaning_mask
            #image[~self.dl1_container.cleaning_mask] = 0

        if self.hillas:

            self.camera_visu.overlay_moments(self.dl2_container.shower, color='r', linewidth=4)

        else:

            self.camera_visu.clear_overlays()

        return np.ma.masked_where(~mask, self.image)

    def press(self, event):

        sys.stdout.flush()

        if event.key == 'enter':
            self.next()

        if event.key == 'right':
            self.set_time(self.time_bin + 1)

        if event.key == 'left':
            self.set_time(self.time_bin - 1)

        if event.key == '+':
            self.set_pixel(self.pixel_id + 1)

        if event.key == '-':
            self.set_pixel(self.pixel_id - 1)

        if event.key == 'h':
            self.axis_next_event_button.set_visible(False)

        if event.key == 'v':
            self.axis_next_event_button.set_visible(True)

        self.update()


class EventClicked:

    def __init__(self, pixel_start):
        self.ind = [0, pixel_start]
