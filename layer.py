import tensorflow as tf


def parameter_count(shape, name=""):
    print "Parametes ", shape, ", Count :", reduce(lambda x, y: x * y, shape), ", Name", name


def weight_variable(shape, name="Weight"):
    parameter_count(shape, name)
    return tf.get_variable(name, shape, initializer=tf.contrib.layers.xavier_initializer_conv2d())


def bias_variable(shape, name="Bias"):
    return tf.get_variable(name, shape, initializer=tf.constant_initializer(.0))


def upscaleBilinear(img_in, method=tf.image.ResizeMethod.BILINEAR, scale=2, align_corners=True):
    return upscaleFlat(img_in, method, scale, align_corners)


def upscaleFlat(img_in, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR, scale=2, align_corners=False):
    target_shape = [img_in.shape[1].value * scale, img_in.shape[2].value * scale]
    return tf.image.resize_images(img_in, target_shape, method=method, align_corners=align_corners)


def max_pool(x, stride=2, padding='SAME'):
    return tf.nn.max_pool(x, ksize=[1, stride, stride, 1], strides=[1, stride, stride, 1], padding=padding)


def resnet_downscale(x, name="resnet_pool"):
    pos = max_pool(tf.nn.relu(x))
    neg = max_pool(tf.nn.relu(-x))
    return tf.concat(values=[pos, neg], axis=3, name=name)


def resnet_upscale(x, name="resnet_upscale"):
    upscaled = upscaleFlat(x)
    sliced = tf.slice(upscaled, [0, 0, 0, 0], [-1, -1, -1, upscaled.shape[3].value / 2])
    return sliced


def avg_pool(x, stride=2, padding='SAME', name="Avg_Pool"):
    return tf.nn.avg_pool(x, ksize=[1, stride, stride, 1], strides=[1, stride, stride, 1], padding=padding, name=name)


def conv(x, layers_in, layers_out, width=6, stride=1, padding='SAME', name="conv"):
    with tf.variable_scope(name):
        w = weight_variable([width, width, layers_in, layers_out])
        b = bias_variable([layers_out])
        return tf.add(tf.nn.conv2d(x, w, strides=[1, stride, stride, 1], padding=padding), b, name=name)


def deconv(x, layers_in, layers_out, width, shape, stride=1, padding='VALID', name="deconv"):
    with tf.variable_scope(name):
        w = weight_variable([width, width, layers_out, layers_in])
        b = bias_variable([layers_out])
        return tf.add(tf.nn.conv2d_transpose(x, filter=w, output_shape=shape, strides=[1, stride, stride, 1], padding=padding), b, name=name)


def drop_conv(keep_prob, x, layers_in, layers_out, width=6, stride=1, padding='SAME', name="drop_conv"):
    with tf.variable_scope(name):
        w = weight_variable([width, width, layers_in, layers_out])
        w = tf.nn.dropout(w, keep_prob) / keep_prob
        b = bias_variable([layers_out])
        return tf.add(tf.nn.conv2d(x, w, strides=[1, stride, stride, 1], padding=padding), b, name=name)


def conv_relu(x, layers_in, layers_out, width=6, stride=1, padding='SAME', name="conv_relu"):
    with tf.variable_scope(name):
        h = conv(x, layers_in, layers_out, width, stride, padding)
        return tf.nn.relu(h, name=name)


def relu_deconv(x, layers_in, layers_out, width, shape, stride=1, padding='VALID', name="relu_deconv"):
    return deconv(tf.nn.relu(x), layers_in, layers_out, width, shape, stride, padding, name)


def batch_normalization(x, training, momentum=0.9, name=None):
    return tf.layers.batch_normalization(x, training=training, momentum=momentum, name=name)


def single_resnet_block(x, layers, width, training, momentum=0.9, name="single_resnet_block"):
    with tf.variable_scope(name):
        result = batch_normalization(x, training=training, momentum=momentum)
        result = tf.nn.relu(result)
        return conv(result, layers, layers, width=width, name=name)


def resnet_block(x, layers, width, training, momentum=0.9, name="resnet_block"):
    with tf.variable_scope((name + "_1")):
        result = single_resnet_block(x,      layers, width, training=training, momentum=momentum)
    with tf.variable_scope((name + "_2")):
        result = single_resnet_block(result, layers, width, training=training, momentum=momentum)
    return tf.add(x, result, name=name)


def resnet_narrow(x, layers, width, training, narrowing=8, momentum=0.9, name="resnet_narrow"):
    with tf.variable_scope(name):
        result = batch_normalization(x, training=training, momentum=momentum)
        result = tf.nn.relu(result)
        result = conv(result, layers, layers / narrowing, width=1, name="narrowing")
        result = tf.nn.relu(result)
        result = conv(result, layers / narrowing, layers / narrowing, width=width, name="conv")
        result = tf.nn.relu(result)
        result = conv(result, layers / narrowing, layers, width=1, name="expand")
        result = tf.nn.relu(result)
        return tf.add(x, result, name=name)


def high_low_noise(value, fraction, name="high_low_noise"):
    with tf.variable_scope(name):
        r = tf.random_uniform([1] + value.shape.as_list()[1:],  minval=0, maxval=1)
        highs = tf.maximum(value, tf.cast(r < fraction / 2, tf.float32) * tf.reduce_max(value))
        high_and_low = tf.maximum(
            tf.ones_like(value) * tf.reduce_min(value),
            (tf.ones_like(value) - tf.cast(r < fraction, tf.float32) * tf.cast(r >= fraction / 2, tf.float32)) * highs)
        return high_and_low


def fully_connected(x, size_in, size_out, name="fully_connected"):
    with tf.variable_scope(name):
        W = weight_variable([size_in, size_out])
        b = bias_variable([size_out])
        return tf.add(tf.matmul(x, W), b, name=name)
