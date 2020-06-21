import tensorflow as tf




class CTCLayer:
    def __init__(self, prev_layer, labels, seq):
        ctc_in_3d_tbc = tf.transpose(prev_layer, [1, 0, 2])

        self.loss = tf.reduce_mean(
            tf.nn.ctc_loss(labels=labels, inputs=ctc_in_3d_tbc, sequence_length=seq,
                           ctc_merge_repeated=True))

        self.decoder = tf.nn.ctc_beam_search_decoder(inputs=ctc_in_3d_tbc, sequence_length=seq,
                                                     beam_width=50, merge_repeated=False)
