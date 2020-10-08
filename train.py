import argparse
import os
import pickle
import time

import gdal
import numpy as np
import tensorflow as tf

from SiamCRNN import SiamCRNN

from util.data_prepro import stad_img

parser = argparse.ArgumentParser()
parser.add_argument('--max_epoch', type=int, default=300, help='epoch to run[default: 50]')
parser.add_argument('--batch_size', type=int, default=1024, help='batch size during training[default: 512]')
parser.add_argument('--learning_rate', type=float, default=2e-4, help='initial learning rate[default: 3e-4]')
parser.add_argument('--save_path', default='model_param', help='model param path')
parser.add_argument('--data_path', default=None, help='dataset path')
parser.add_argument('--gpu_num', type=int, default=1, help='number of GPU to train')

# basic params
FLAGS = parser.parse_args()

BATCH_SZ = FLAGS.batch_size
LEARNING_RATE = FLAGS.learning_rate
MAX_EPOCH = FLAGS.max_epoch
SAVE_PATH = FLAGS.save_path
DATA_PATH = FLAGS.data_path
GPU_NUM = FLAGS.gpu_num
BATCH_PER_GPU = BATCH_SZ // GPU_NUM


class ChangeTrainer(object):

    def __init__(self):
        self.Input_X = None
        self.Input_Y = None
        self.label = None
        self.is_training = None

        self.net = None
        self.pred = None
        self.loss = None
        self.opt = None
        self.train_op = None
        self.global_step = tf.Variable(0, trainable=False)
        self.siamcrnn_model = SiamCRNN()


    def load_data(self):
        data_set_X = gdal.Open('data/GF_2_2/0411')  # data set X
        data_set_Y = gdal.Open('data/GF_2_2/0901')  # data set Y

        img_width = data_set_X.RasterXSize  # image width
        img_height = data_set_X.RasterYSize  # image height

        img_X = data_set_X.ReadAsArray(0, 0, img_width, img_height)
        img_Y = data_set_Y.ReadAsArray(0, 0, img_width, img_height)

        img_X = stad_img(img_X)  # (C, H, W)
        img_Y = stad_img(img_Y)
        img_X = np.transpose(img_X, [1, 2, 0])  # (H, W, C)
        img_Y = np.transpose(img_Y, [1, 2, 0])  # (H, W, C)
        return img_X, img_Y

    def training(self):
        train_X, train_Y, train_label = self._load_train_data(path=DATA_PATH)
        self.valid_X, self.valid_Y, self.valid_label = self._load_valid_data(path=DATA_PATH)
        train_label = np.reshape(train_label, (-1, 1))
        self.valid_label = np.reshape(self.valid_label, (-1, 1))
        self.valid_sz = self.valid_X.shape[0]

        shape_1 = train_X.shape
        shape_2 = train_Y.shape
        train_sz = train_X.shape[0]
        self.Input_X = tf.placeholder(dtype=tf.float32, shape=[None, shape_1[1], shape_1[2], shape_1[3]],
                                      name='Input_X')
        self.Input_Y = tf.placeholder(dtype=tf.float32, shape=[None, shape_2[1], shape_2[2], shape_2[3]],
                                      name='Input_Y')
        self.label = tf.placeholder(dtype=tf.float32, shape=[None, 1])
        self.is_training = tf.placeholder(dtype=tf.bool, name='is_training')

       
        self.net, self.pred, _, _ = self.siamcrnn_model.get_model(Input_X=self.Input_X, Input_Y=self.Input_Y,
                                                                  data_format='NHWC',
                                                                  is_training=self.is_training)  # (B, 2)
        self.loss = self._get_loss(label=self.label, logits=self.net)
        self.opt = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE)
        self.train_op = self.opt.minimize(loss=self.loss)
        best_loss = 100000
        iter_in_epoch = train_sz // BATCH_SZ
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        saver = tf.train.Saver(max_to_keep=0, var_list=tf.global_variables())
        total_time = 0
        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())
            epoch_sz = MAX_EPOCH

            for epoch in range(epoch_sz):
                tic = time.time()
                ave_loss = 0
                train_idx = np.arange(0, train_sz)
                np.random.shuffle(train_idx)
                for _iter in range(iter_in_epoch):
                    start_idx = _iter * BATCH_SZ
                    end_idx = (_iter + 1) * BATCH_SZ
                    batch_train_X = train_X[train_idx[start_idx:end_idx]]
                    batch_train_Y = train_Y[train_idx[start_idx:end_idx]]
                    batch_label = train_label[train_idx[start_idx:end_idx]]
                    loss, _, logits = sess.run(
                        [self.loss, self.train_op, self.net],
                        feed_dict={
                            self.Input_X: batch_train_X,
                            self.Input_Y: batch_train_Y,
                            self.label: batch_label,
                            self.is_training: True
                        })
                    ave_loss += loss
                ave_loss /= iter_in_epoch
                toc = time.time()
                total_time += (toc - tic)
                # print("epoch %d , loss is %f take %.3f s , min logits is %.3f, min pred is %.3f" % (
                #     epoch + 1, ave_loss, time.time() - tic, min_logits, min_pred))
                val_loss = self.evaluate(sess)

                if (epoch + 1) % 5 == 0:
                    if val_loss < best_loss:
                        best_loss = val_loss
                        _path = saver.save(sess, os.path.join(SAVE_PATH, "best_model.ckpt"))
                        print("best model is saved")
                    _path = saver.save(sess, os.path.join(SAVE_PATH, "cha_model_%d.ckpt" % (epoch + 1)))
                    print("epoch %d, model saved in file: " % (epoch + 1), _path)
                    # self.evaluate(sess)
            _path = saver.save(sess, os.path.join(SAVE_PATH, 'final_model.ckpt'))
            print("Model saved in file: ", _path)
        print(total_time)

    def evaluate(self, sess):
        iter_in_epoch = self.valid_sz // BATCH_SZ
        valid_idx = np.arange(0, self.valid_sz)
        np.random.shuffle(valid_idx)
        ave_loss = 0
        for _iter in range(iter_in_epoch):
            start_idx = _iter * BATCH_SZ
            end_idx = (_iter + 1) * BATCH_SZ
            batch_valid_X = self.valid_X[valid_idx[start_idx:end_idx]]
            batch_valid_Y = self.valid_Y[valid_idx[start_idx:end_idx]]
            batch_label = self.valid_label[valid_idx[start_idx:end_idx]]
            loss = sess.run(self.loss, feed_dict={  # (B, 2)
                self.Input_X: batch_valid_X,
                self.Input_Y: batch_valid_Y,
                self.label: batch_label,
                self.is_training: False
            })
            ave_loss += loss
        ave_loss /= iter_in_epoch
        print("evaluate is done, validation loss is %.3f" % ave_loss)
        return ave_loss

    def _get_loss(self, label, logits):
        loss = tf.reduce_mean(
            tf.nn.weighted_cross_entropy_with_logits(targets=label, logits=logits, pos_weight=5, name='weight_loss'))
        return loss

    def _load_train_data(self, path):
        with open(os.path.join(path, 'train_sample_X.pickle'), 'rb') as file:
            train_X = pickle.load(file)
        with open(os.path.join(path, 'train_sample_Y.pickle'), 'rb') as file:
            train_Y = pickle.load(file)
        with open(os.path.join(path, 'train_label.pickle'), 'rb') as file:
            train_label = pickle.load(file)

        return train_X, train_Y, train_label

    def _load_valid_data(self, path):
        with open(os.path.join(path, 'valid_sample_X.pickle'), 'rb') as file:
            valid_X = pickle.load(file)
        with open(os.path.join(path, 'valid_sample_Y.pickle'), 'rb') as file:
            valid_Y = pickle.load(file)
        with open(os.path.join(path, 'valid_label.pickle'), 'rb') as file:
            valid_label = pickle.load(file)

        return valid_X, valid_Y, valid_label


if __name__ == '__main__':
    trainer = ChangeTrainer()
    trainer.training()
