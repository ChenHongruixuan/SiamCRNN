import tensorflow as tf
from tensorflow.layers import batch_normalization

_EPSILON = 1e-7


def conv_2d(inputs, kernel_size, output_channel, stride, name, is_reuse=False, padding='SAME', data_format='NHWC',
            is_bn=False, is_training=True, activation=None):
    """
    2D Conv Layer
    :param name: scope name
    :param inputs: (B, H, W, C)
    :param kernel_size: [kernel_h, kernel_w]
    :param output_channel: feature num
    :param stride: a list of 2 ints
    :param padding: type of padding, str
    :param data_format: str, the format of input data
    :param is_bn: bool, is batch normalization
    :param is_training: bool, is training
    :param activation: activation function, such as tf.nn.relu
    :return: outputs
    """
    with tf.variable_scope(name) as scope:
        if is_reuse:
            scope.reuse_variables()
        kernel_h, kernel_w = kernel_size
        stride_h, stride_w = stride
        if data_format == 'NHWC':
            kernel_shape = [kernel_h, kernel_w, inputs.get_shape()[-1].value, output_channel]
        else:
            kernel_shape = [kernel_h, kernel_w, inputs.get_shape()[1].value, output_channel]
        init = tf.keras.initializers.he_normal()
        kernel = tf.get_variable(name='conv_kenel', shape=kernel_shape, initializer=init, dtype=tf.float32)
        # kernel = tf.Variable(tf.truncated_normal(kernel_shape, dtype=tf.float32, stddev=0.1))
        outputs = tf.nn.conv2d(input=inputs,
                               filter=kernel,
                               strides=[1, stride_h, stride_w, 1], padding=padding,
                               data_format=data_format)
        biases = tf.Variable(tf.constant(0.1, shape=[output_channel], dtype=tf.float32))
        outputs = outputs + biases
        if is_bn:
            outputs = batch_normalization(outputs, training=is_training)
        if activation is not None:
            outputs = activation(outputs)
        return outputs


def conv_2d_transpose(inputs, kernel_size, output_channel, output_shape, stride, name, padding='SAME',
                      data_format='NHWC', is_bn=False, is_training=True, activation=None):
    """
    2D Transpose Conv Layer
    :param output_shape:
    :param name: scope name
    :param inputs: (B, H, W, C)
    :param kernel_size: [kernel_h, kernel_w]
    :param output_channel: feature num
    :param stride: a list of 2 ints
    :param padding: type of padding, str
    :param data_format: str, the format of input data
    :param is_bn: bool, is batch normalization
    :param is_training: bool, is training
    :param activation: activation function, such as tf.nn.relu
    :return: outputs
    """
    with tf.variable_scope(name) as scope:
        kernel_h, kernel_w = kernel_size
        stride_h, stride_w = stride
        if data_format == 'NHWC':
            kernel_shape = [kernel_h, kernel_w, output_channel, inputs.get_shape()[-1].value]
        else:
            kernel_shape = [kernel_h, kernel_w, output_channel, inputs.get_shape()[1].value]

        init = tf.keras.initializers.he_normal()
        kernel = tf.get_variable(name='conv_kenel', shape=kernel_shape, initializer=init, dtype=tf.float32)

        # calculate output shape
        # batch_size, height, width, _ = inputs.get_shape().as_list()
        # out_height = get_deconv_dim(height, stride_h, kernel_h, padding)
        # out_width = get_deconv_dim(width, stride_w, kernel_w, padding)
        # output_shape = [batch_size, out_height, out_width, output_channel]

        # kernel = tf.Variable(tf.truncated_normal(kernel_shape, dtype=tf.float32, stddev=0.1))
        outputs = tf.nn.conv2d_transpose(value=inputs,
                                         filter=kernel,
                                         output_shape=output_shape,
                                         strides=[1, stride_h, stride_w, 1], padding=padding,
                                         data_format=data_format)
        biases = tf.Variable(tf.constant(0.1, shape=[output_channel], dtype=tf.float32))
        outputs = outputs + biases
        if is_bn:
            outputs = batch_normalization(outputs, training=is_training)
        if activation is not None:
            outputs = activation(outputs)
        return outputs


# from slim.convolution2d_transpose
def get_deconv_dim(dim_size, stride_size, kernel_size, padding):
    dim_size *= stride_size

    if padding == 'VALID' and dim_size is not None:
        dim_size += max(kernel_size - stride_size, 0)
    return dim_size


def max_pool_2d(inputs, kernel_size, stride, padding='SAME'):
    """
    2D Max Pool Layer
    :param inputs: (B, H, W, C)
    :param kernel_size: [kernel_h, kernel_w]
    :param padding: type of padding, str
    :return: outputs
    """
    kernel_h, kernel_w = kernel_size
    stride_h, stride_w = stride
    outputs = tf.nn.max_pool(inputs,
                             ksize=[1, kernel_h, kernel_w, 1],
                             strides=[1, stride_h, stride_w, 1],
                             padding=padding)
    return outputs


def avg_pool_2d(inputs, kernel_size, stride, padding='SAME'):
    """
    2D Avg Pool Layer
    :param inputs: (B, H, W, C)
    :param kernel_size: [kernel_h, kernel_w]
    :param padding: type of padding, str
    :return: outputs
    """
    kernel_h, kernel_w = kernel_size
    stride_h, stride_w = stride
    outputs = tf.nn.avg_pool(inputs,
                             ksize=[1, kernel_h, kernel_w, 1],
                             strides=[1, stride_h, stride_w, 1],
                             padding=padding)
    return outputs


def fully_connected(inputs, num_outputs, is_training=True, is_bn=False, activation=None):
    """
    Fully connected layer with non-linear operation
    :param inputs: (B, N)
    :param num_outputs: int
    :param is_training: bool
    :param is_bn: bool
    :param activation: activation function, such as tf.nn.relu
    :return: outputs: (B, num_outputs)
    """

    num_input_units = inputs.get_shape()[-1].value
    weights = tf.Variable(tf.truncated_normal([num_input_units, num_outputs], dtype=tf.float32, stddev=0.1))
    outputs = tf.matmul(inputs, weights)
    biases = tf.Variable(tf.constant(0.1, shape=[num_outputs], dtype=tf.float32))
    outputs = tf.nn.bias_add(outputs, biases)
    if is_bn:
        outputs = batch_normalization(outputs, training=is_training)
    if activation is not None:
        outputs = activation(outputs)
    return outputs


def weight_binary_cross_entropy(target, output, weight=1.0, from_logits=False):
    """weight binary crossentropy between an output tensor and a target tensor.

        # Arguments
            target: A tensor with the same shape as `output`.
            output: A tensor.
            from_logits: Whether `output` is expected to be a logits tensor.
                By default, we consider that `output`
                encodes a probability distribution.

        # Returns
            A tensor.
        """
    # Note: tf.nn.sigmoid_cross_entropy_with_logits
    # expects logits, Keras expects probabilities.
    if not from_logits:
        # transform back to logits
        _epsilon = _to_tensor(epsilon(), output.dtype.base_dtype)
        output = tf.clip_by_value(output, _epsilon, 1 - _epsilon)
        output = tf.log(output / (1 - output))

    return tf.nn.weighted_cross_entropy_with_logits(targets=target,
                                                    logits=output,
                                                    pos_weight=weight)


def _to_tensor(x, dtype):
    """Convert the input `x` to a tensor of type `dtype`.

    # Arguments
        x: An object to be converted (numpy array, list, tensors).
        dtype: The destination type.

    # Returns
        A tensor.
    """
    return tf.convert_to_tensor(x, dtype=dtype)


def epsilon():
    """Returns the value of the fuzz factor used in numeric expressions.

    # Returns
        A float.

    # Example
    ```python
        >>> keras.backend.epsilon()
        1e-07
    ```
    """
    return _EPSILON
