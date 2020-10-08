import tensorflow as tf
from net_util import conv_2d, max_pool_2d, avg_pool_2d, fully_connected


class SiamCRNN(object):

    def get_model(self, Input_X, Input_Y, data_format='NHWC', is_training=True):
        net_X = self._feature_extract_layer(inputs=Input_X, name='Fea_Ext_',
                                            data_format=data_format,
                                            is_training=is_training)
        net_Y = self._feature_extract_layer(inputs=Input_Y, name='Fea_Ext_',
                                            data_format=data_format,
                                            is_training=is_training, is_reuse=True)

        fea_1 = tf.squeeze(net_X, axis=1)
        fea_2 = tf.squeeze(net_Y, axis=1)
       
        logits, pred = self._change_judge_layer(feature_1=fea_1, feature_2=fea_2, name='Cha_Jud_',
                                                is_training=is_training)
        return logits, pred

    def _feature_extract_layer(self, inputs, name='Fea_Ext_', data_format='NHWC', is_training=True, is_reuse=False):
        with tf.variable_scope(name) as scope:
            if is_reuse:
                scope.reuse_variables()
            # (B, H, W, C) --> (B, H, W, 16)
            layer_1 = conv_2d(inputs=inputs, kernel_size=[3, 3], output_channel=16, stride=[1, 1], name='layer_1_conv',
                              padding='SAME', data_format=data_format, is_training=is_training, is_bn=False,
                              activation=tf.nn.relu)
            layer_2 = conv_2d(inputs=layer_1, kernel_size=[3, 3], output_channel=16, stride=[1, 1], name='layer_2_conv',
                              padding='SAME', data_format=data_format, is_training=is_training, is_bn=False,
                              activation=tf.nn.relu)

            layer_2 = tf.contrib.layers.dropout(inputs=layer_2, is_training=is_training, keep_prob=0.8)

            # (B, H/2, W/2, 16) --> (B, H/2, W/2, 32)
            layer_3 = conv_2d(inputs=layer_2, kernel_size=[3, 3], output_channel=32, stride=[1, 1], padding='SAME',
                              name='layer_3_conv', data_format=data_format, is_training=is_training, is_bn=False,
                              activation=tf.nn.relu)
            layer_4 = conv_2d(inputs=layer_3, kernel_size=[3, 3], output_channel=32, stride=[1, 1], padding='SAME',
                              name='layer_4_conv', data_format=data_format, is_training=is_training, is_bn=False,
                              activation=tf.nn.relu)
            layer_4 = tf.contrib.layers.dropout(inputs=layer_4, is_training=is_training, keep_prob=0.7)

            # # (B, H/2, W/2, 32) --> (B, H/2, W/2, 64)
            layer_5 = conv_2d(inputs=layer_4, kernel_size=[3, 3], output_channel=64, stride=[1, 1], padding='SAME',
                              name='layer_5_conv', data_format=data_format, is_training=is_training, is_bn=False,
                              activation=tf.nn.relu)
            net = conv_2d(inputs=layer_5, kernel_size=[5, 5], output_channel=64, stride=[1, 1], padding='VALID',
                          name='layer_6_conv', data_format=data_format, is_training=is_training, is_bn=False,
                          activation=tf.nn.relu)
            net = tf.contrib.layers.dropout(inputs=net, is_training=is_training, keep_prob=0.5)
            return net

    def _change_judge_layer(self, feature_1, feature_2, name='Cha_Jud_', is_training=True,
                            activation=tf.nn.sigmoid):
        with tf.variable_scope(name) as scope:
            seq = tf.concat([feature_1, feature_2], axis=1)  # (B, 2, 128)
            num_units = [128, 64]
            cells = [tf.nn.rnn_cell.LSTMCell(num_unit, activation=tf.nn.tanh) for num_unit in num_units]
            mul_cells = tf.nn.rnn_cell.MultiRNNCell(cells)
            output, cell_state = tf.nn.dynamic_rnn(mul_cells, seq, dtype=tf.float32, time_major=False)
            hidden_state = tf.contrib.layers.dropout(inputs=output[:, -1], is_training=is_training, keep_prob=0.5)
            logits_0 = fully_connected(hidden_state, num_outputs=32, is_training=is_training, is_bn=False,
                                       activation=tf.nn.tanh)
            logits = fully_connected(logits_0, num_outputs=1, is_training=is_training, is_bn=False)
            pred = activation(logits)
            return logits, pred
