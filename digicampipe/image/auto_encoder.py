from pkg_resources import resource_filename
from decimal import Decimal, ROUND_HALF_EVEN
import numpy as np
import os

from matplotlib import pyplot as plt
from cts_core.camera import Camera
import tensorflow as tf
from astropy.io import fits

from digicampipe.io.event_stream import event_stream
from digicampipe.calib.camera import filter, r1, random_triggers
from digicampipe.utils import geometry
from digicampipe.image.cones_image import get_pixel_nvs

__all__ = [
    "CameraData",
    "AutoEncoder",
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
            print_every=500
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
        self._fits = fits.open(filename)
        self.data = self._fits[0].data
        self.shape = self.data.shape

    def __del__(self):
        if self._fits is not None:
            self._fits.close()

    def _new_event_stream(self, datafiles_list):
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
        if self.flags is not None:
            events_stream = filter.filter_event_types(
                events_stream,
                flags=self.flags
            )
        if self.min_adc is None:
            return events_stream
        wanted_pixels = np.arange(len(self.digicam.Pixels))
        if self.unwanted_pixels is not None:
            mask = np.ones_like(wanted_pixels, dtype=bool)
            mask[self.unwanted_pixels] = False
            wanted_pixels = wanted_pixels[mask]
        for event in events_stream:
            tel = event.r0.tels_with_data[0]
            r0 = event.r0.tel[tel]
            adc_samples = r0.adc_samples[wanted_pixels, :]
            baseline = np.reshape(r0.digicam_baseline[wanted_pixels], (-1, 1))
            if np.any(adc_samples - baseline > self.min_adc):
                yield event

    def create_fits_file(self, filename, datafiles_list,
                         print_every=500):
        events_stream = self._new_event_stream(datafiles_list)
        nvs = CameraData.get_pixels_pos_in_skew_base(
            self.digicam_config_file
        )
        n_pixel_u, n_pixel_v = (np.max(nvs, axis=0) + 1).tolist()
        n_event, n_sample = self.get_data_size(datafiles_list)
        if os.path.isfile(filename):
            os.remove(filename)
        with open(filename, 'ab+') as file:
            hdr = fits.Header()
            hdr['SIMPLE'] = (True, 'conforms to FITS standard')
            hdr['BITPIX'] = (16, 'array data type')  # dtype=int16
            hdr['NAXIS'] = (4, 'number of array dimensions')
            hdr['NAXIS1'] = n_pixel_u
            hdr['NAXIS2'] = n_pixel_v
            hdr['NAXIS3'] = n_sample
            hdr['NAXIS4'] = n_event
            hdr['EXTEND'] = True
            shdu = fits.StreamingHDU(file, hdr)
            event_loaded = 0
            for event in events_stream:
                tel = event.r0.tels_with_data[0]
                r0 = event.r0.tel[tel]
                adc_samples = r0.adc_samples
                baseline = r0.digicam_baseline
                data_event = np.zeros([n_pixel_u, n_pixel_v, n_sample],
                                      dtype=np.int16)
                for i, nv in enumerate(nvs):
                    if i not in self.unwanted_pixels:
                        data_event[nv[0], nv[1], :] = adc_samples[i, :] - \
                                                      baseline[i]
                event_loaded += 1
                shdu.write(data_event)
                if event_loaded % print_every == 0:
                    print(event_loaded, "events loaded")
            print('closing stream after', event_loaded, "events loaded")
            events_stream.close()
            shdu.close()

    def get_data_size(self, datafiles_list):
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
        plt.ion()
        first = True
        plt.figure()
        for i, data_event in enumerate(self.data):
            n_sample = data_event.shape[2]
            for t in range(n_sample):
                title_text = 'event: %i / %i\nt: %i / %i ns' % \
                             (i + 1, len(self.data), (t + 1) * 4, n_sample * 4)
                if first:
                    img = plt.imshow(data_event[:, :, t],
                                     vmin=-100, vmax=100, cmap='seismic')
                    first = False
                    plt.colorbar()
                    title = plt.title(title_text)
                else:
                    img.set_array(data_event[:, :, t])
                    title.set_text(title_text)
                plt.pause(.02)
        plt.ioff()

    def hist_adc(self):
        plt.ioff()
        plt.figure()
        max_data = np.max(self.data, axis=(1, 2, 3))
        plt.hist(max_data,100, log=True)
        plt.show()


class AutoEncoder(object):
    sess = None
    data_stream = None

    def __init__(self, camera_data, model_path=None,
                 kernel_size=[5, 5, 5], n_out=1024):
        self.data = camera_data
        tf.reset_default_graph()
        self.sess = tf.Session()
        n_event, h_size, v_size, t_size = self.data.shape
        self.x = tf.placeholder(tf.float32, [None, h_size, v_size, t_size])
        self.is_train = tf.placeholder(tf.bool)
        self.x_encoded = self._create_model_encoder(self.x, n_out=n_out,
                                                    kernel_size=kernel_size)
        self.x_decoded = self._create_model_decoder(self.x_encoded,
                                                    kernel_size=kernel_size)
        losses = tf.losses.mean_squared_error(self.x, self.x_decoded)
        self.loss = tf.reduce_mean(losses)
        self.saver = tf.train.Saver()
        if model_path is not None:
            self.saver.restore(self.sess, model_path)

    def __del__(self):
        if self.sess is not None:
            self.sess.close()

    def _create_model_encoder(self, x, n_out=1024, kernel_size=[5, 5, 5]):
        n_event, h_size, v_size, t_size = self.data.shape
        with tf.variable_scope("encoder"):
            x_flat = tf.reshape(x, [-1, h_size*v_size*t_size])
            bn1_flat = tf.layers.batch_normalization(x_flat,
                                                     training=self.is_train)
            bn1 = tf.reshape(bn1_flat, [-1, h_size, v_size, t_size, 1])
            conv1 = tf.layers.conv3d(bn1, filters=64, kernel_size=kernel_size,
                                     strides=[2, 2, 2], activation=tf.nn.relu,
                                     padding='same')
            bn2 = tf.layers.batch_normalization(conv1, training=self.is_train)
            conv2 = tf.layers.conv3d(bn2, filters=64, kernel_size=kernel_size,
                                     strides=[2, 2, 2], activation=tf.nn.relu,
                                     padding='same')
            bn3 = tf.layers.batch_normalization(conv2, training=self.is_train)
            conv3 = tf.layers.conv3d(bn3, filters=64, kernel_size=kernel_size,
                                     strides=[2, 2, 2], activation=tf.nn.relu,
                                     padding='same')
            conv3_flat = tf.reshape(conv3,
                                    [-1, int(h_size/8*v_size/8*t_size/8)])
            bn4 = tf.layers.batch_normalization(conv3_flat,
                                                training=self.is_train)
            dense1 = tf.layers.dense(bn4, n_out, activation=tf.nn.relu)
            return dense1

    def _create_model_decoder(self, z, kernel_size=[5, 5, 5]):
        n_event, h_size, v_size, t_size = self.data.shape
        with tf.variable_scope("decoder"):
            dense1 = tf.layers.dense(z, int(h_size*v_size*t_size/8/8/8),
                                     activation=tf.nn.relu)
            bn1_flat = tf.layers.batch_normalization(dense1,
                                                     training=self.is_train)
            bn1 = tf.reshape(bn1_flat, [-1, int(h_size / 8),
                                        int(v_size / 8), int(t_size / 8), 64])
            conv1 = tf.layers.conv3d_transpose(
                bn1, filters=64, kernel_size=kernel_size,
                strides=[2, 2, 2], activation=tf.nn.relu, padding='same')
            bn2 = tf.layers.batch_normalization(conv1,
                                                training=self.is_train)
            conv2 = tf.layers.conv3d_transpose(
                bn2, filters=64, kernel_size=kernel_size,
                strides=[2, 2, 2], activation=tf.nn.relu, padding='same')
            bn3 = tf.layers.batch_normalization(conv2,
                                                training=self.is_train)
            conv3 = tf.layers.conv3d_transpose(bn3, filters=1,
                                               kernel_size=kernel_size,
                                               strides=[2, 2, 2],
                                               padding='same')
            x_decoded = tf.reshape(conv3, [-1, h_size, v_size, t_size])
            return x_decoded

    def train(self, model_path, n_epoch=20, batch_size=100, print_every=100,
              learning_rate=1e-2, n_sample=None):
        optimizer = tf.train.AdamOptimizer(learning_rate)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_step = optimizer.minimize(self.loss)
        self.sess.run(tf.global_variables_initializer())
        n_event, h_size, v_size, t_size = self.data.shape
        max_iter = int(n_event * n_epoch / batch_size)
        losses_train = []
        losses_val = []
        iter_val = []
        for it in range(max_iter):
            batch_x = self.data.get_batch(batch_size,
                                          n_sample=n_sample,
                                          type_set='train')
            _, loss = self.sess.run(
                [train_step, self.loss],
                feed_dict={self.x: batch_x, self.is_train: True}
            )
            losses_train.append(loss)
            if it % print_every == 0:
                batch_x_val = self.data.get_batch(
                    batch_size,
                    n_sample=n_sample,
                    type_set='val'
                )
                loss_val = self.sess.run(
                    self.loss,
                    feed_dict={self.x: batch_x_val, self.is_train: False}
                )
                print('iter', it+1, '/', max_iter, ',',
                      'loss=', loss, ' validation_loss=', loss_val)
                losses_val.append(loss_val)
                iter_val.append(it)
        self.saver.save(self.sess, model_path)
        print('model saved in', model_path)
        return losses_train, losses_val, iter_val


# create fits file
datafile_crab = [
    '/mnt/sst1m/raw/2017/10/30/SST1M01/SST1M01_0_000.%03d.fits.fz'
    % i for i in range(11,92)
]
camera_data = resource_filename(
    'digicampipe',
    os.path.join('tests', 'resources', 'camera_data.fits')
)
pixel_not_wanted = [
    1038, 1039, 1002, 1003, 1004, 966, 967, 968, 930, 931, 932, 896,
    1085, 1117, 1118, 1119, 1120, 1146, 1147, 1148, 1149, 1150, 1151,
    1152, 1172, 1173, 1174, 1175, 1176, 1177, 1178, 1179, 1180, 1181,
    1196, 1197, 1198, 1199, 1200, 1201, 1202, 1203, 1204, 1205, 1206,
    1219, 1220, 1221, 1222, 1223, 1224, 1225, 1226, 1239, 1240, 1241,
    1242, 1243, 1256, 1257]
data = CameraData(camera_data, datafiles_list=datafile_crab,
                  digicam_config_file=digicam_config_file_default,
                  unwanted_pixels=pixel_not_wanted, 
                  flags=[1],
                  min_adc=100,
                  print_every=500)

# load fits file:
# data = CameraData(camera_data)
# show data:
# data.animate()
data.hist_adc()

"""
# training:
model_path = resource_filename(
    'digicampipe',
    os.path.join('tests', 'resources', 'ae_conv3d_512.ckpt')
)
ae = AutoEncoder(data, kernel_size=[3, 3, 3], n_out=512)
loss_history, val_loss, val_iters = ae.train(
    model_path, n_epoch=20, batch_size=35, print_every=10,
    learning_rate=1e-3, n_sample=None
)
plt.plot(loss_history, 'b-')
plt.plot(val_iters, val_loss, 'r-')
plt.xlabel('iterations')
plt.ylabel('loss')
plt.show()

# loading
# ae = AutoEncoder(data, model_path=model_path, kernel_size=[3, 3, 3], nout=1024)
"""