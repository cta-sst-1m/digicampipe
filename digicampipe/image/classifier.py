import os
import time
import sys

from matplotlib import use as mpl_use
mpl_use('Agg')
from matplotlib import pyplot as plt
import numpy as np
import tensorflow as tf

from digicampipe.image.camera_data import CameraData

__all__ = ["Classifier",]


class Classifier(object):
    sess = None
    data_stream = None

    def __init__(self, classes_data, classes=['gamma', 'proton'],
                 model_path=None, kernel_size=(5, 5, 5), n_sample=None,
                 regularizer_scale=0.1, n_filter=128):
        """
        Create a classifier.
        :param classes_data: list of CameraData, each one corresponding to a
        given class
        :param classes: name of the classes in the same order as in
        classes_data
        :param model_path: if given, the model at the given path is loaded.
        :param kernel_size: size of the convolutions kernels (3 elements)
        :param n_sample: number of sample to consider in each events
        :param regularizer_scale: factor used for weights regularization
        :param n_filter: max number of filters in the convolutions
        """
        self.classes = classes
        n_class = len(self.classes)
        self.classes_dict = {}
        for i, key in enumerate(classes):
            self.classes_dict[key] = i
        self.classes_data = classes_data
        tf.reset_default_graph()
        self.sess = tf.Session()
        event_shape = None
        for data in self.classes_data:
            if event_shape is None:
                event_shape = data.event_shape
            elif event_shape != data.event_shape:
                raise AttributeError('classes_data with incompatible shapes')
        self.h_size, self.v_size, t_size = event_shape
        if n_sample is None:
            self.t_size = t_size
        else:
            self.t_size = n_sample
        self.kernel_size = kernel_size
        self.n_filter = int(n_filter)
        print('\n####### MODEL CREATION #######\n')
        print('kernel size:', kernel_size,
              't_size:', self.t_size)
        # input data to the classifier
        self.x = tf.placeholder(
            tf.float32,
            [None, self.h_size, self.v_size, self.t_size]
        )
        # true labels
        self.label = tf.placeholder(
            tf.int32,
            [None]
        )
        self.is_train = tf.placeholder(tf.bool)
        if regularizer_scale > 0:
            self.regularizer = tf.contrib.layers.l2_regularizer(
                scale=regularizer_scale
            )
        else:
            self.regularizer = None
        self.logits = self._create_model(self.x, n_class)
        one_hot = tf.one_hot(self.label, n_class)
        print('one_hot:', one_hot.shape)
        self.loss_class = tf.losses.softmax_cross_entropy(
            one_hot, self.logits,
            reduction=tf.losses.Reduction.SUM_BY_NONZERO_WEIGHTS
        )
        self.loss_reg = tf.reduce_sum(
            tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        )
        self.loss = self.loss_class + self.loss_reg
        self.scores = tf.nn.softmax(self.logits)
        self.label_predictions = tf.argmax(self.logits, axis=1)
        self.saver = tf.train.Saver()
        if model_path is not None:
            self.saver.restore(self.sess, model_path)
            print("model", model_path, "restored.")

    def __del__(self):
        if self.sess is not None:
            self.sess.close()

    def _create_model(self, x, n_class):
        h_size, v_size, t_size = self.h_size, self.v_size, self.t_size
        n_filter = self.n_filter
        with tf.variable_scope("classifier"):
            x = tf.reshape(x, [-1, h_size, v_size, t_size, 1])
            conv1 = tf.layers.conv3d(x, filters=n_filter // 4,
                                     kernel_size=self.kernel_size,
                                     strides=[1, 1, 1],
                                     activation=tf.nn.relu,
                                     padding='same',
                                     kernel_regularizer=self.regularizer,
                                     name='conv1')
            maxpool1 = tf.layers.max_pooling3d(
                conv1, pool_size=(2, 2, 2), strides=(2, 2, 2), name='pool1'
            )
            conv2 = tf.layers.conv3d(maxpool1, filters=n_filter // 2,
                                     kernel_size=self.kernel_size,
                                     strides=[1, 1, 1],
                                     activation=tf.nn.relu,
                                     padding='same',
                                     kernel_regularizer=self.regularizer,
                                     name='conv2')
            maxpool2 = tf.layers.max_pooling3d(
                conv2, pool_size=(2, 2, 2), strides=(2, 2, 2), name='pool2'
            )
            conv3 = tf.layers.conv3d(maxpool2, filters=n_filter // 1,
                                     kernel_size=self.kernel_size,
                                     strides=[1, 1, 1],
                                     activation=tf.nn.relu,
                                     padding='same',
                                     kernel_regularizer=self.regularizer,
                                     name='conv3')
            maxpool3 = tf.layers.max_pooling3d(
                conv3, pool_size=(2, 2, 2), strides=(2, 2, 2), name='pool3'
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
                maxpool3,
                [-1, int(h_size / 8 * v_size / 8 * t_size / 8 * n_filter)]
            )
            print('maxpool4_flat:', maxpool4_flat.shape)
            logits = tf.layers.dense(maxpool4_flat, n_class,
                                     kernel_regularizer=self.regularizer)
            print('logits:', logits.shape)
            return logits

    def get_batch(self, batch_size=100, class_batch=None, type_set='train'):
        n_class = len(self.classes)
        if class_batch == None:
            # randomize the classes contents of the batch
            class_batch = np.random.choice(n_class, size=batch_size)
        data = np.zeros([batch_size, self.h_size,
                         self.v_size, self.t_size])
        for i in range(n_class):
            # get batch of each class and put it in the right place
            pos_class_in_batch = np.where(class_batch == i)[0]
            class_amount = len(pos_class_in_batch)
            events_class = self.classes_data[i].get_batch(
                class_amount, n_sample=self.t_size, type_set=type_set
            )
            data[pos_class_in_batch, :, :, :] = events_class['data']
        return data, class_batch

    def train(self, model_path, n_epoch=20, batch_size=100, print_every=100,
              learning_rate=1e-2):
        optimizer = tf.train.AdamOptimizer(learning_rate)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_step = optimizer.minimize(self.loss)
        self.sess.run(tf.global_variables_initializer())
        n_event = 0
        for data in self.classes_data:
            n_event += data.n_event
        max_iter = int(float(n_event * n_epoch) / batch_size)

        print('\n######## TRAINING ##########\n')
        print('running for', max_iter, 'iterations with',
              batch_size, 'events per iteration (over ', n_epoch,
              'epochs).')
        print('learning rate:', learning_rate)

        losses_train = []
        accuracies_train = []
        losses_val = []
        iter_val = []
        accuracies_val = []
        for it in range(max_iter):
            data, classes_train = self.get_batch(batch_size)
            # training
            _, loss, loss_class, loss_reg, label = self.sess.run(
                [train_step, self.loss, self.loss_class, self.loss_reg,
                 self.label_predictions],
                feed_dict={self.x: data, self.label: classes_train,
                           self.is_train: True}
            )
            accuracy = np.sum(classes_train == label) / batch_size
            losses_train.append(loss)
            accuracies_train.append(accuracy)
            del data
            if it % print_every == 0:
                data_val, classes_val = self.get_batch(
                    batch_size,
                    type_set='val'
                )
                loss_val, label_val = self.sess.run(
                    [self.loss, self.label_predictions],
                    feed_dict={self.x: data_val, self.label: classes_val,
                           self.is_train: False}
                )
                accuracy = np.sum(classes_val == label_val) / batch_size
                print('iter', it + 1, '/', max_iter, ',',
                      'loss=%.1f (%.1f+%.1f)' % (loss, loss_class, loss_reg),
                      '\tvalidation_loss=%.1f' % loss_val,
                      '\taccuracy=%.1f%%' % accuracy*100)
                losses_val.append(loss_val)
                iter_val.append(it)
                accuracies_val.append(accuracy)
                del data_val
        self.saver.save(self.sess, model_path)
        print('model saved in', model_path)
        return losses_train, accuracies_train, losses_val, iter_val, \
            accuracies_val


def train(kernel_size=(3, 3, 3), learning_rate=1e-2,
          batch_size=250, n_epoch=20, regularizer_scale=0.1, n_filter=64,
          plot_loss=True):
    gamma_file = os.path.join('/home/reniery/prog/digicampipe/autoencoder',
                              'gamma_data_mc.fits')
    proton_file = os.path.join('/home/reniery/prog/digicampipe/autoencoder',
                               'proton_data_mc.fits')
    # load fits file:
    gamma_data = CameraData(gamma_file)
    proton_data = CameraData(proton_file)

    kernel_size_str = '%dx%dx%d' % (
    kernel_size[0], kernel_size[1], kernel_size[2])
    opt_str = kernel_size_str + '_' + '_' + str(
        learning_rate) + \
              '_' + str(batch_size) + '_' + str(n_epoch) + '_' + \
              str(regularizer_scale) + '_' + str(n_filter)

    model_path = os.path.join(
        '/home/reniery/prog/digicampipe/autoencoder/models',
        'classifier_conv3d_' + opt_str + '.ckpt'
    )

    classifier = Classifier(
        [gamma_data, proton_data],
        classes=['gamma', 'proton'],
        kernel_size=kernel_size,
        n_filter=n_filter,
        n_sample=48,
        regularizer_scale=regularizer_scale
    )
    t_start = time.clock()
    (losses_train, accuracies_train,
     losses_val, iter_val, accuracies_val) = classifier.train(
        model_path, n_epoch=n_epoch, batch_size=batch_size, print_every=10,
        learning_rate=learning_rate
    )
    t_spent = time.clock() - t_start
    print('training took', t_spent, 's')
    np.savez('models/classifier_loss_' + opt_str + '.npz',
             losses_train=losses_train,
             accuracies_train=accuracies_train,
             losses_val=losses_val,
             iter_val=iter_val,
             accuracies_val=accuracies_val,
             t_spent=t_spent)
    if plot_loss:
        fig, ax1 = plt.subplots()
        ax1.plot(losses_train, 'b-')
        ax1.plot(iter_val, losses_val, 'b--')
        ax1.ylim([1, 1e4])
        ax1.xlabel('iterations')
        ax1.ylabel('loss', color='b')
        ax1.tick_params('y', colors='b')
        ax2 = ax1.twinx()
        ax2.plot(accuracies_train, 'r-')
        ax2.plot(iter_val, accuracies_val, 'r--')
        ax2.ylim([1, 1e4])
        ax2.xlabel('')
        ax2.ylabel('accuracy (%)', color='r')
        ax1.tick_params('y', colors='r')
        plt.savefig('models/classifier_loss_' + opt_str + '.png')


def main():
    kernel_size = (4, 4, 4)
    learning_rate = 1e-3
    batch_size = 50
    n_epoch = 20
    regularizer_scale = 1e-3
    n_filter = 128
    print('\n######## MAIN ##########\n')
    print('kernel_size=', kernel_size)
    print('learning_rate=', learning_rate)
    print('batch_size=', batch_size)
    print('n_epoch=', n_epoch)
    print('regularizer_scale=', regularizer_scale)
    print('n_filter=', n_filter)
    train(kernel_size=kernel_size, learning_rate=learning_rate,
          batch_size=batch_size, n_epoch=n_epoch, plot_loss=True,
          regularizer_scale=regularizer_scale, n_filter=n_filter)


if __name__ == '__main__':
    main()
