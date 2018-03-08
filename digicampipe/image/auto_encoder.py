from pkg_resources import resource_filename
import os

from matplotlib import pyplot as plt
import tensorflow as tf

from digicampipe.image.camera_data import CameraData

__all__ = [
    "AutoEncoder",
]


class AutoEncoder(object):
    sess = None
    data_stream = None

    def __init__(self, camera_data, model_path=None,
                 kernel_size=(5, 5, 5), n_out=1024):
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

    def _create_model_encoder(self, x, n_out=1024, kernel_size=(5, 5, 5)):
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

    def _create_model_decoder(self, z, kernel_size=(5, 5, 5)):
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


camera_data = resource_filename(
    'digicampipe',
    os.path.join('data', 'camera_data.fits')
)

# load fits file:
data = CameraData(camera_data)

# training:
model_path = resource_filename(
    'digicampipe',
    os.path.join('tests', 'resources', 'ae_conv3d_512.ckpt')
)
ae = AutoEncoder(data, kernel_size=(3, 3, 3), n_out=512)
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
