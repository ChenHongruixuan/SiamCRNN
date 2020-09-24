import cv2 as cv

import gdal
import numpy as np
import tensorflow as tf
from util.data_prepro import stad_img

from SiamCRNN import SiamCRNN
import time



def load_data(path_X, path_Y):
    data_set_X = gdal.Open(path_X)  # data set X
    data_set_Y = gdal.Open(path_Y)  # data set Y

    img_width = data_set_X.RasterXSize  # image width
    img_height = data_set_X.RasterYSize  # image height

    img_X = data_set_X.ReadAsArray(0, 0, img_width, img_height)
    img_Y = data_set_Y.ReadAsArray(0, 0, img_width, img_height)

    img_X = stad_img(img_X)  # (C, H, W)
    img_Y = stad_img(img_Y)
    img_X = np.transpose(img_X, [1, 2, 0])  # (H, W, C)
    img_Y = np.transpose(img_Y, [1, 2, 0])  # (H, W, C)
    return img_X, img_Y


def infer_result():
    patch_sz = 5
    batch_size = 1000

    img_X, img_Y = load_data()
    img_X = np.pad(img_X, ((2, 2), (2, 2), (0, 0)), 'constant')
    img_Y = np.pad(img_Y, ((2, 2), (2, 2), (0, 0)), 'constant')
    img_height, img_width, channel = img_X.shape  # image width
    
    edge = patch_sz // 2
    sample_X = []
    sample_Y = []
    for i in range(edge, img_height - edge):
        for j in range(edge, img_width - edge):
            sample_X.append(img_X[i - edge:i + edge + 1, j - edge:j + edge + 1, :])
            sample_Y.append(img_Y[i - edge:i + edge + 1, j - edge:j + edge + 1, :])
    sample_X = np.array(sample_X)
    sample_Y = np.array(sample_Y)

    epoch = sample_X.shape[0] // batch_size

    Input_X = tf.placeholder(dtype=tf.float32, shape=[None, patch_sz, patch_sz, channel], name='Input_X')
    Input_Y = tf.placeholder(dtype=tf.float32, shape=[None, patch_sz, patch_sz, channel], name='Input_Y')
    is_training = tf.placeholder(dtype=tf.bool, name='is_training')

    model_path = 'model_param'
    model = SiamCRNN()
    net, result = model.get_model(Input_X=Input_X, Input_Y=Input_Y, data_format='NHWC', is_training=is_training)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    saver = tf.train.Saver()
    logits_result_list = []
    pred_results_list = []
    path = None

    with tf.Session() as sess:
        ckpt = tf.train.get_checkpoint_state(model_path)
        if ckpt and ckpt.model_checkpoint_path:
            path = ckpt.model_checkpoint_path
            saver.restore(sess, ckpt.model_checkpoint_path)
        tic = time.time()
        for _epoch in range(100):
            pred_result = sess.run([result], feed_dict={
                Input_X: sample_X[batch_size * _epoch:batch_size * (_epoch + 1)],
                Input_Y: sample_Y[batch_size * _epoch:batch_size * (_epoch + 1)],
                is_training: False
            })
            pred_results_list.append(pred_result)
  
    pred = np.reshape(pred_results_list, (img_height, img_width))

    idx_1 = (pred <= 0.5)
    idx_2 = (pred > 0.5)
    pred[idx_1] = 0
    pred[idx_2] = 255
    cv.imwrite('SiamCRNN.bmp', pred)



if __name__ == '__main__':
    infer_result()