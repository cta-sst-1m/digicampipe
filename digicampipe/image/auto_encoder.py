import os
import time
import sys

from matplotlib import use as mpl_use
mpl_use('Agg')
from matplotlib import pyplot as plt
import numpy as np
import tensorflow as tf

from digicampipe.image.camera_data import CameraData

__all__ = [
    "AutoEncoder",
]


class AutoEncoder(object):
    sess = None
    data_stream = None

    def __init__(self, camera_data, model_path=None,
                 kernel_size=(5, 5, 5), n_out=1024, n_sample=None, 
                 regularizer_scale=0.1, n_filter=128):
        self.data = camera_data
        tf.reset_default_graph()
        self.sess = tf.Session()
        _, self.h_size, self.v_size, t_size = self.data.event_shape
        if n_sample is None:
            self.t_size = t_size
        else:
            self.t_size = n_sample
        self.kernel_size = kernel_size
        self.n_out = int(n_out)
        self.n_filter = int(n_filter)
        print('\n####### MODEL CREATION #######\n')
        print('kernel size:', kernel_size, 'n_out:', n_out, 't_size:', 
              self.t_size)
        self.x = tf.placeholder(
            tf.float32, 
            [None, self.h_size, self.v_size, self.t_size]
        )
        self.is_train = tf.placeholder(tf.bool)
        if regularizer_scale > 0:
            self.regularizer = tf.contrib.layers.l2_regularizer(
                scale=regularizer_scale
            )
        else:
            self.regularizer = None
        digicam_config_file = self.data.digicam_config_file
        self.x_encoded = self._create_model_encoder(self.x)
        self.x_decoded = self._create_model_decoder(self.x_encoded)
        x_norms = tf.norm(self.x, axis=3, keepdims=True)
        mask_data = tf.stop_gradient(x_norms > 0)
        print("x=", self.x.shape)
        print("x_decoded=", self.x_decoded.shape)
        print("mask_data=", mask_data.shape)
        self.x_decoded = tf.multiply(self.x_decoded, 
                                     tf.cast(mask_data, tf.float32))
        diff = self.x - self.x_decoded
        """
        norm_order = 2
        self.losses = tf.norm(diff, ord=norm_order , axis=[1,2,3], )
        self.losses /= tf.reduce_sum(
            tf.cast(self.mask_loss, tf.float32), 
            axis=[1, 2]
        )**(1/norm_order)
        loss_reco = tf.reduce_mean(self.losses)
        """
        weights_raw = tf.maximum(self.x, 20)
        weights_unorm = tf.multiply(weights_raw, tf.cast(mask_data, tf.float32))
        weights = weights_unorm/tf.reduce_sum(weights_unorm, axis=[1, 2, 3], 
                                              keepdims=True)
        self.weights = tf.stop_gradient(weights)
        self.loss_reco = tf.sqrt(tf.losses.mean_squared_error(
            self.x, 
            self.x_decoded, 
            weights=self.weights,
            reduction=tf.losses.Reduction.MEAN
        ))
        self.loss_reg = tf.reduce_sum(
            tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        )
        self.loss = self.loss_reco + self.loss_reg

        self.saver = tf.train.Saver()
        if model_path is not None:
            self.saver.restore(self.sess, model_path)
            print("model", model_path, "restored.")

    def __del__(self):
        if self.sess is not None:
            self.sess.close()

    def _create_model_encoder(self, x):
        h_size, v_size, t_size = self.h_size, self.v_size, self.t_size
        n_filter = self.n_filter
        with tf.variable_scope("encoder"):
            x = tf.reshape(x, [-1, h_size, v_size, t_size, 1])
            conv1 = tf.layers.conv3d(x, filters=n_filter//4, 
                                     kernel_size=self.kernel_size,
                                     strides=[1, 1, 1],
                                     activation=tf.nn.relu,
                                     padding='same',
                                     kernel_regularizer=self.regularizer,
                                     name='conv1')
            maxpool1 = tf.layers.max_pooling3d(
                conv1, pool_size=(2,2,2), strides=(2,2,2), name='pool1'
            )
            conv2 = tf.layers.conv3d(maxpool1, filters=n_filter//2, 
                                     kernel_size=self.kernel_size,
                                     strides=[1, 1, 1],
                                     activation=tf.nn.relu,
                                     padding='same',
                                     kernel_regularizer=self.regularizer,
                                     name='conv2')
            maxpool2 = tf.layers.max_pooling3d(
                conv2, pool_size=(2,2,2), strides=(2,2,2), name='pool2'
            )
            conv3 = tf.layers.conv3d(maxpool2, filters=n_filter//1, 
                                     kernel_size=self.kernel_size,
                                     strides=[1, 1, 1],
                                     activation=tf.nn.relu,
                                     padding='same',
                                     kernel_regularizer=self.regularizer,
                                     name='conv3')
            maxpool3 = tf.layers.max_pooling3d(
                conv3, pool_size=(2,2,2), strides=(2,2,2), name='pool3'
            )
#            conv4 = tf.layers.conv3d(maxpool3, filters=n_filter, 
#                                     kernel_size=self.kernel_size,
#                                     strides=[1, 1, 1],
#                                     activation=tf.nn.relu,
#                                     padding='same',
#                                     kernel_regularizer=self.regularizer,
#                                     name='conv4')
#            maxpool4 = tf.layers.max_pooling3d(
#                conv4, pool_size=(2,2,2), strides=(2,2,2), name='pool4')
#            print('maxpool4:',maxpool4.shape)
            maxpool4_flat = tf.reshape(
                maxpool3, [-1, int(h_size/8*v_size/8*t_size/8*n_filter)]
            )
            print('maxpool4_flat:',maxpool4_flat.shape)            
            dense1 = tf.layers.dense(maxpool4_flat, self.n_out,
                                     kernel_regularizer=self.regularizer)
            print('dense1:',dense1.shape)
            return dense1

    def _create_model_decoder(self, z):
        h_size, v_size, t_size = self.h_size, self.v_size, self.t_size
        n_filter = self.n_filter
        with tf.variable_scope("decoder"):
            dense1_flat = tf.layers.dense(
                z, int((h_size/8+3)*(v_size/8+3)*(t_size/8+3)*n_filter),
                activation=tf.nn.relu,
                kernel_regularizer=self.regularizer)
            dense1 = tf.reshape(
                dense1_flat, 
                [-1, int(h_size/8+3), int(v_size/8+3), int(t_size/8+3), n_filter]
            )
            print('dense1:',dense1.shape)
            #conv1 = tf.layers.conv3d_transpose(
            #    dense1, filters=n_filter//1, 
            #    kernel_size=self.kernel_size,
            #    strides=[2, 2, 2], 
            #    activation=tf.nn.relu, 
            #    padding='same',
            #    kernel_regularizer=self.regularizer
            #)
            #conv1 = tf.slice(conv1, [0, 1, 1, 1, 0], [-1,  int(h_size/8+3), int(v_size/8+3), int(t_size/8+3), n_filter])
            #print('conv1:',conv1.shape)
            conv2 = tf.layers.conv3d_transpose(
                dense1, filters=n_filter//1,
                kernel_size=self.kernel_size,
                strides=[2, 2, 2], 
                activation=tf.nn.relu, 
                padding='same',
                kernel_regularizer=self.regularizer
            )
            conv2 = tf.slice(conv2, [0, 1, 1, 1, 0], [-1,  int(h_size/4+2), int(v_size/4+2), int(t_size/4+2), n_filter//1])
            print('conv2:',conv2.shape)
            conv3 = tf.layers.conv3d_transpose(
                conv2, filters=n_filter//2,
                kernel_size=self.kernel_size,
                strides=[2, 2, 2],
                activation=tf.nn.relu, 
                padding='same',
                kernel_regularizer=self.regularizer
            )
            conv3 = tf.slice(conv3, [0, 1, 1, 1, 0], [-1,  int(h_size/2+1), int(v_size/2+1), int(t_size/2+1), n_filter//2])
            print('conv3:',conv3.shape)
            conv4 = tf.layers.conv3d_transpose(
                conv3, filters=1,
                kernel_size=self.kernel_size,
                strides=[2, 2, 2], 
                padding='same',
                kernel_regularizer=self.regularizer
            )
            print('conv4:',conv4.shape)
            conv4 = tf.slice(conv4, [0, 1, 1, 1, 0], [-1,  h_size, v_size, t_size, 1])
            x_decoded = tf.reshape(conv4, [-1, h_size, v_size, t_size])
            return x_decoded

    def train(self, model_path, n_epoch=20, batch_size=100, print_every=100,
              learning_rate=1e-2):
        optimizer = tf.train.AdamOptimizer(learning_rate)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_step = optimizer.minimize(self.loss)
        self.sess.run(tf.global_variables_initializer())
        n_event = self.data.n_event
        max_iter = int(float(n_event * n_epoch) / batch_size)

        print('\n######## TRAINING ##########\n')
        print('running for', max_iter, 'iterations with', 
              batch_size, 'events per iteration (over ', n_epoch, 'epochs).')
        print('learning rate:', learning_rate)

        losses_train = []
        losses_val = []
        iter_val = []
        for it in range(max_iter):
            events = self.data.get_batch(batch_size,
                                         n_sample=self.t_size,
                                         type_set='train')
            _, loss, loss_reco, loss_reg = self.sess.run(
                [train_step, self.loss, self.loss_reco, self.loss_reg],
                feed_dict={self.x: events['data'], self.is_train: True}
            )
            losses_train.append(loss)
            del events
            if it % print_every == 0:
                events_val = self.data.get_batch(
                    batch_size,
                    n_sample=self.t_size,
                    type_set='val'
                )
                loss_val = self.sess.run(
                    self.loss,
                    feed_dict={self.x: events_val['data'],
                               self.is_train: False}
                )
                print('iter', it+1, '/', max_iter, ',',
                      'loss=%.1f (%.1f+%.1f)'% (loss, loss_reco, loss_reg), 
                      '\tvalidation_loss=%.1f' % loss_val)
                losses_val.append(loss_val)
                iter_val.append(it)
                del events_val
        self.saver.save(self.sess, model_path)
        print('model saved in', model_path)
        return losses_train, losses_val, iter_val

    def encode_decode(self, x):
        x_decoded, x_encoded, loss, weights = self.sess.run(
            [self.x_decoded, self.x_encoded, self.loss, self.weights],
            feed_dict={self.x: x, self.is_train: False}
        )
        batch_size = x.shape[0]
        assert(x_encoded.shape == (batch_size, self.n_out))
        return x_decoded, loss, weights


def train(kernel_size=(3, 3, 3), n_out=512, learning_rate=1e-2, 
        batch_size=250, n_epoch=20, regularizer_scale=0.1, n_filter=64, 
        plot_loss=True):
    camera_data = os.path.join('/home/reniery/prog/digicampipe/autoencoder',
                               'camera_data_test.fits')
    unwanted_pixels = [
        1038, 1039, 1002, 1003, 1004, 966, 967, 968, 930, 931, 932, 896,
        1085, 1117, 1118, 1119, 1120, 1146, 1147, 1148, 1149, 1150, 1151,
        1152, 1172, 1173, 1174, 1175, 1176, 1177, 1178, 1179, 1180, 1181,
        1196, 1197, 1198, 1199, 1200, 1201, 1202, 1203, 1204, 1205, 1206,
        1219, 1220, 1221, 1222, 1223, 1224, 1225, 1226, 1239, 1240, 1241,
        1242, 1243, 1256, 1257
    ]
    # load fits file:
    data = CameraData(camera_data, unwanted_pixels=unwanted_pixels)

    kernel_size_str = '%dx%dx%d' % (kernel_size[0],kernel_size[1],kernel_size[2])
    opt_str = kernel_size_str + '_' + str(n_out) + '_' + str(learning_rate) +\
              '_' + str(batch_size) + '_' + str(n_epoch) + '_' +\
              str(regularizer_scale) + '_' + str(n_filter)

    model_path = os.path.join('/home/reniery/prog/digicampipe/autoencoder',
                              'models',
                              'ae_conv3d_' + opt_str + '.ckpt')

    ae = AutoEncoder(data, kernel_size=kernel_size, n_out=n_out, n_sample=48,
                     regularizer_scale=regularizer_scale, n_filter=n_filter)
    t_start = time.clock()
    losses_train, losses_val, iter_val = ae.train(
        model_path, n_epoch=n_epoch, batch_size=batch_size, print_every=10,
        learning_rate=learning_rate
    )
    t_spent = time.clock() - t_start
    print('training took', t_spent, 's')
    np.savez('models/loss_'+ opt_str + '.npz',
             losses_train=losses_train,
             losses_val=losses_val,
             iter_val=iter_val,
             t_spent=t_spent)
    if plot_loss:
        plt.plot(losses_train, 'b-')
        plt.semilogy(iter_val, losses_val, 'r-')
        plt.ylim([1, 1e4])
        plt.xlabel('iterations')
        plt.ylabel('loss')
        plt.savefig('models/loss_'+ opt_str + '.png')


def main():
    if len(sys.argv) != 8:
        print('usage:', sys.argv[0], 'kernel_size', 'n_out', 'learning_rate',
              'batch_size', 'n_epoch', 'regularizer_scale', 'n_filter')
        exit()
    kernel_size_str = sys.argv[1]
    kernel_size = tuple(int(s) for s in kernel_size_str.split('x'))
    n_out = int(sys.argv[2])
    learning_rate = float(sys.argv[3])
    batch_size = int(sys.argv[4])
    n_epoch = int(sys.argv[5])
    regularizer_scale = float(sys.argv[6])
    n_filter = int(sys.argv[7])
    print('\n######## MAIN ##########\n')
    print('kernel_size_str=', kernel_size_str)
    print('n_out=', n_out)
    print('learning_rate=', learning_rate)
    print('batch_size=', batch_size)
    print('n_epoch=', n_epoch)
    print('regularizer_scale=', regularizer_scale)
    print('n_filter=', n_filter)
    train(kernel_size=kernel_size, n_out=n_out, learning_rate=learning_rate, 
          batch_size=batch_size, n_epoch=n_epoch, plot_loss=True,
          regularizer_scale=regularizer_scale, n_filter=n_filter)
    

if __name__ == '__main__':
    main()
