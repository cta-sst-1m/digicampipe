from pkg_resources import resource_filename
from decimal import Decimal, ROUND_HALF_EVEN
import numpy as np
import os
import time

from matplotlib import pyplot as plt
from cts_core.camera import Camera
from astropy.io import fits
from astropy import units as u
from ctapipe.image.hillas import hillas_parameters
from scipy.interpolate import LinearNDInterpolator

from digicampipe.io.event_stream import event_stream
from digicampipe.calib.camera import filter, r0, r1, dl0, dl1, dl2, random_triggers
from digicampipe.utils import geometry
from digicampipe.utils import utils
from digicampipe.image.cones_image import get_pixel_nvs

__all__ = [
    "CameraData",
    "animate",
]

datafile_default = resource_filename(
    'digicampipe',
    os.path.join(
        'tests',
        'resources',
        'SST1M01_0_000.072.fits.fz'
    )
)
digicam_config_file_default = resource_filename(
    'digicampipe',
    os.path.join(
        'tests',
        'resources',
        'camera_config.cfg'
    )
)


def animate(data):
    plt.ion()
    first = True
    plt.figure()
    for i, data_event in enumerate(data):
        n_sample = data_event.shape[2]
        for t in range(n_sample):
            title_text = 'event: %i / %i\nt: %i / %i ns' % \
                         (i + 1, len(data), (t + 1) * 4, n_sample * 4)
            if first:
                img = plt.imshow(data_event[:, :, t],
                                 vmin=-500, vmax=500, cmap='seismic')
                first = False
                plt.colorbar()
                title = plt.title(title_text)
            else:
                img.set_array(data_event[:, :, t])
                title.set_text(title_text)
            plt.pause(.02)
            if t == 10:
                plt.pause(5)
    plt.ioff()


class CameraData(object):
    _fits = None

    def __init__(
            self,
            filename,
            datafiles_list=None,
            digicam_config_file=digicam_config_file_default,
            unwanted_pixels=None,
            flags=None,
            min_adc=None,
            print_every=100
    ):
        self.digicam_config_file = digicam_config_file
        self.digicam = Camera(_config_file=self.digicam_config_file)
        self.geo = geometry.generate_geometry_from_camera(camera=self.digicam)
        self.unwanted_pixels = unwanted_pixels
        self.flags = flags
        self.min_adc = min_adc
        if datafiles_list is not None:
            self.create_fits_file(filename, datafiles_list,
                                  print_every=print_every)
        self._fits = fits.open(filename, memmap=True)
        # print('shape of data from file', self._fits[0].data.shape)
        self.data = self._fits[0].data
        # print('shape of data', self.data.shape)
        self.shape = self.data.shape
        data_shuffle_indexes = np.random.permutation(self.data.shape[0])
        self.data = self.data[data_shuffle_indexes, :, :, :]

    def __del__(self):
        if self._fits is not None:
            self._fits.close()

    def _new_event_stream(self, datafiles_list):
        time_integration_options = {'mask': None,
                                    'mask_edges': None,
                                    'peak': None,
                                    'window_start': 3,
                                    'window_width': 7,
                                    'threshold_saturation': np.inf,
                                    'n_samples': 50,
                                    'timing_width': 6,
                                    'central_sample': 11}
        peak_position = utils.fake_timing_hist(
            time_integration_options['n_samples'],
            time_integration_options['timing_width'],
            time_integration_options['central_sample'])

        (
            time_integration_options['peak'],
            time_integration_options['mask'],
            time_integration_options['mask_edges']
        ) = utils.generate_timing_mask(
            time_integration_options['window_start'],
            time_integration_options['window_width'],
            peak_position
        )
        additional_mask = np.ones(1296)
        additional_mask[self.unwanted_pixels] = 0
        additional_mask = additional_mask > 0
        picture_threshold = 15
        boundary_threshold = 10
        shower_distance = 200 * u.mm
        events_stream = event_stream(
            file_list=datafiles_list,
            camera_geometry=self.geo,
            camera=self.digicam,
            expert_mode=True,
        )
        if self.unwanted_pixels is not None:
            events_stream = filter.set_pixels_to_zero(
                events_stream,
                unwanted_pixels=self.unwanted_pixels
            )
        events_stream = random_triggers.fill_baseline_r0(events_stream, n_bins=100)
        #events_stream = r0.fill_baseline_r0(events_stream, unwanted_pixels=self.unwanted_pixels)
        # Stop events that are not triggered by DigiCam algorithm (end of clocked triggered events)
        if self.flags is not None:
            events_stream = filter.filter_event_types(
                events_stream,
                flags=self.flags
            )
        events_stream = filter.filter_missing_baseline(events_stream)
        events_stream = r1.calibrate_to_r1(events_stream, None)
        # Run the dl0 calibration (data reduction, does nothing)
        events_stream = dl0.calibrate_to_dl0(events_stream)
        # Run the dl1 calibration (compute charge in photons + cleaning)
        events_stream = dl1.calibrate_to_dl1(events_stream,
                                           time_integration_options,
                                           additional_mask=additional_mask,
                                           picture_threshold=picture_threshold,
                                           boundary_threshold=boundary_threshold)
        """
        # Return only showers with total number of p.e. above min_photon
        events_stream = filter.filter_shower(
            events_stream, min_photon=args['--min_photon'])
        """
        # Run the dl2 calibration (Hillas)
        events_stream = dl2.calibrate_to_dl2(
            events_stream, reclean=True, shower_distance=shower_distance)
        wanted_pixels = np.arange(len(self.digicam.Pixels))
        if self.unwanted_pixels is not None:
            mask = np.ones_like(wanted_pixels, dtype=bool)
            mask[self.unwanted_pixels] = False
            wanted_pixels = wanted_pixels[mask]
        for event in events_stream:
            tel = event.r0.tels_with_data[0]
            r0_cont = event.r0.tel[tel]
            adc_samples = r0_cont.adc_samples[wanted_pixels, :]
            baseline = np.reshape(r0_cont.digicam_baseline[wanted_pixels], (-1, 1))
            if self.min_adc is None:
                yield event
            if np.any(adc_samples - baseline > self.min_adc):
                yield event

    def create_fits_file(self, filename, datafiles_list,
                         print_every=500):
        nvs = CameraData.get_pixels_pos_in_skew_base(
            self.digicam_config_file
        )
        n_pixel_u, n_pixel_v = (np.max(nvs, axis=0) + 1).tolist()
        pix_pos = np.array([self.geo.pix_x, self.geo.pix_y]).transpose()
        image_x, image_y = (np.linspace(-504, 504, 48).reshape(-1,1), np.linspace(-504, 504, 48))
        n_event, n_sample = self.get_data_size(datafiles_list)
        if os.path.isfile(filename):
            os.remove(filename)
        events_stream = self._new_event_stream(datafiles_list)
        n_unwanted_pixel = len(self.unwanted_pixels)
        wanted_pixels = []
        for i in range(pix_pos.shape[0]):
            if i not in self.unwanted_pixels:
                wanted_pixels.append(i)
        with open(filename, 'ab+') as file:
            hdr = fits.Header()
            hdr['SIMPLE'] = (True, 'conforms to FITS standard')
            hdr['BITPIX'] = (16, 'array data type')  # dtype=int16
            hdr['NAXIS'] = (4, 'number of array dimensions')
            hdr['NAXIS1'] = n_sample
            hdr['NAXIS2'] = n_pixel_u
            hdr['NAXIS3'] = n_pixel_v
            hdr['NAXIS4'] = n_event
            hdr['EXTEND'] = True
            shdu = fits.StreamingHDU(file, hdr)
            event_loaded = 0
            for event in events_stream:
                tel = event.r0.tels_with_data[0]
                r0 = event.r0.tel[tel]
                adc_samples = r0.adc_samples
                hillas = event.dl2.shower
                psi = hillas.psi.rad
                rot_angle = psi + np.pi/2
                cen_x = u.Quantity(hillas.cen_x).value
                cen_y = u.Quantity(hillas.cen_y).value
                baseline = r0.digicam_baseline
                data_event = np.zeros([n_pixel_u, n_pixel_v, n_sample],
                                      dtype=np.int16)
                pix_pos_translate = pix_pos - np.array([cen_x, cen_y])
                pix_pos_transform = pix_pos_translate.dot(
                    np.array([[np.cos(rot_angle), np.sin(-rot_angle)], [np.sin(rot_angle), np.cos(rot_angle)]])
                )
                for t in range(n_sample):
                    f_2d_interp = LinearNDInterpolator(
                        np.vstack((pix_pos_transform[wanted_pixels], pix_pos_transform[self.unwanted_pixels])),
                        np.hstack((adc_samples[wanted_pixels, t] - baseline[wanted_pixels],
                                   np.zeros((n_unwanted_pixel,)))),
                        fill_value=0)
                    data_event[:, :, t] = f_2d_interp(image_x, image_y)
                event_loaded += 1
                shdu.write(data_event)
                if event_loaded % print_every == 0:
                    print(event_loaded, "events loaded")
            print('closing stream after', event_loaded, "events loaded")
            events_stream.close()
            shdu.close()

    def get_data_size(self, datafiles_list):
        print("getting size of the event stream ...")
        events_stream = self._new_event_stream(datafiles_list)
        event = next(events_stream)
        tel = event.r0.tels_with_data[0]
        r0 = event.r0.tel[tel]
        adc_samples = r0.adc_samples
        n_pixels, n_samples = adc_samples.shape
        size = 1
        for _ in events_stream:
            size += 1
        events_stream.close()
        print("got", size, 'events with', n_samples, 'samples')
        return size, n_samples

    @staticmethod
    def get_pixels_pos_in_skew_base(camera_config_file):
        pixel_nvs = get_pixel_nvs(digicam_config_file=camera_config_file)
        pixel_nvs = np.array(pixel_nvs)
        nvs_from_orig = pixel_nvs - np.min(pixel_nvs, axis=1, keepdims=True)
        precision = Decimal('0.1')
        nvs_dec = np.array([
            [
                Decimal(n1*3).quantize(precision, rounding=ROUND_HALF_EVEN)/3,
                Decimal(n2*3).quantize(precision, rounding=ROUND_HALF_EVEN)/3,
            ]
            for n1, n2 in nvs_from_orig.transpose()
        ])
        nvs = nvs_dec.astype(int)
        return nvs

    def get_used_pixel_in_skew_base(self):
        nvs = self.get_pixels_pos_in_skew_base(self.digicam_config_file)
        nvs[self.unwanted_pixels]
        mask_shape =  np.max(nvs, axis=0)
        mask = np.zeros(mask_shape+1)
        for i, (x, y) in enumerate(nvs):
            if self.unwanted_pixels is not None and i in self.unwanted_pixels:
                continue
            mask[x, y] = 1
        return mask

    def get_batch(self, batch_size, n_sample=None, type_set='train'):
        n_event = self.data.shape[0]
        n_train = int(0.8 * n_event)
        n_val = int(0.1 * n_event)
        n_test = n_event - n_train - n_val
        train_data, val_data, test_data = np.split(self.data,
                                                   [n_train, n_train+n_val],
                                                   axis=0)
        if type_set == 'train':
            data = train_data
            n_event = n_train
        elif type_set == 'val':
            data = val_data
            n_event = n_val
        elif type_set == 'test':
            print('WARNING: test data should only be used once !')
            data = test_data
            n_event = n_test
        else:
            print('ERROR: unknown type_set:', type_set)
            return
        if n_sample is None:
            n_sample = data.shape[-1]
        if batch_size > n_event:
            print('WARNING: the given batch size is too big for the dataset')
        picked_events = np.random.permutation(n_event)[:batch_size]
        return data[picked_events, :, :, :n_sample]

    def animate(self):
        animate(self.data)

    def hist_adc(self):
        plt.ioff()
        plt.figure()
        max_data = np.max(self.data, axis=(1, 2, 3))
        plt.hist(max_data,100, log=True)
        plt.show()


def create_file(camera_data, datafiles_list=None, print_every=100):
    # create fits file
    datafile_crab = [
        '/sst1m/raw/2017/10/30/SST1M01/SST1M01_20171030.%03d.fits.fz'
        % i for i in range(11,92)
    ]
    pixel_not_wanted = [
        1038, 1039, 1002, 1003, 1004, 966, 967, 968, 930, 931, 932, 896,
        1085, 1117, 1118, 1119, 1120, 1146, 1147, 1148, 1149, 1150, 1151,
        1152, 1172, 1173, 1174, 1175, 1176, 1177, 1178, 1179, 1180, 1181,
        1196, 1197, 1198, 1199, 1200, 1201, 1202, 1203, 1204, 1205, 1206,
        1219, 1220, 1221, 1222, 1223, 1224, 1225, 1226, 1239, 1240, 1241,
        1242, 1243, 1256, 1257
    ]
    if datafiles_list is None:
        datafiles_list = datafile_crab
    CameraData(
        camera_data, datafiles_list=datafiles_list,
        digicam_config_file=digicam_config_file_default,
        unwanted_pixels=pixel_not_wanted,
        flags=[1],
        min_adc=100,
        print_every=print_every
    )


def show_file(camera_data):
    # load fits file:
    data = CameraData(camera_data)
    # show data:
    data.animate()
    data.hist_adc()


if __name__ == '__main__':
    camera_data = os.path.join('/home/reniery/prog/digicampipe/autoencoder',
                               'camera_data_test.fits')
    #create_file(camera_data,
    #            print_every=10)
    show_file(camera_data)
