import tensorflow as tf



class RNNLayer:
    def __init__(self, prev_layer):
        """create RNN layers and return output of these layers"""
        rnn_in_3d = tf.squeeze(prev_layer, axis=[2])

        # basic cells which is used to build RNN
        num_hidden = 256
        cells = [tf.contrib.rnn.LSTMCell(num_units=num_hidden, state_is_tuple=True) for _ in range(2)]  # 2 layers

        # stack basic cells
        stacked = tf.contrib.rnn.MultiRNNCell(cells, state_is_tuple=True)

        # bidirectional RNN
        # BxTxF -> BxTx2H
        ((fw, bw), _) = tf.nn.bidirectional_dynamic_rnn(cell_fw=stacked, cell_bw=stacked, inputs=rnn_in_3d, dtype=rnn_in_3d.dtype)

        # BxTxH + BxTxH -> BxTx2H -> BxTx1X2H
        concat = tf.expand_dims(tf.concat([fw, bw], 2), 2)

        # project output to chars (including blank): BxTx1x2H -> BxTx1xC -> BxTxC
        kernel = tf.Variable(tf.truncated_normal([1, 1, num_hidden * 2, 80], stddev=0.1))
        self.out = tf.squeeze(tf.nn.atrous_conv2d(value=concat, filters=kernel, rate=1, padding='SAME'), axis=[2])


