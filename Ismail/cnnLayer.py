import tensorflow as tf


class CNNLayer:
    def __init__(self, input_imgs, is_train):
        cnn_in_4d = tf.expand_dims(input=input_imgs, axis=3)

        # list of parameters for the layers
        kernel_vals = [5, 5, 3, 3, 3]
        feature_vals = [1, 32, 64, 128, 128, 256]
        stride_vals = pool_vals = [(2, 2), (2, 2), (1, 2), (1, 2), (1, 2)]
        num_layers = len(stride_vals)

        # create layers
        pool = cnn_in_4d  # input to first CNN layer
        for i in range(num_layers):
            kernel = tf.Variable(
                tf.truncated_normal([kernel_vals[i], kernel_vals[i], feature_vals[i], feature_vals[i + 1]], stddev=0.1))
            conv = tf.nn.conv2d(pool, kernel, padding='SAME', strides=(1, 1, 1, 1))
            conv_norm = tf.layers.batch_normalization(conv, training=is_train)
            relu = tf.nn.relu(conv_norm)
            pool = tf.nn.max_pool(relu, (1, pool_vals[i][0], pool_vals[i][1], 1),
                                  (1, stride_vals[i][0], stride_vals[i][1], 1), 'VALID')

        self.out = pool


