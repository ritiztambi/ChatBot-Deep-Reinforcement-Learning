import numpy as np
import tensorflow as tf

from tensorflow.contrib import rnn


class NeuralConversationModel():
    def __init__(self, conf, vocab):
        self.conf = conf
        self.vocab = vocab

    def add_place_holder(self):
        self.encoder_input_sequence = tf.placeholder(
            dtype=tf.int32, shape=(None, None), name='encoder_event')
        self.decoder_input_sequence = tf.placeholder(
            dtype=tf.int32, shape=(None, None), name='decoder_input_event')
        self.decoder_output_sequence = tf.placeholder(
            dtype=tf.int32, shape=(None, None), name='decoder_output_event')
        self.encoder_input_length = tf.placeholder(
            dtype=tf.int32, shape=[None], name='encoder_sequence_length')
        self.decoder_input_length = tf.placeholder(
            dtype=tf.int32, shape=[None], name='decoder_sequence_length')

    def add_embeddings(self):
        #(batch,sequence_length,embedding_size)
        with tf.device('/cpu:0'):
            self.embedding_matrix = tf.get_variable(
                name="embedding_matrix", shape=(self.conf.vocab_size, self.conf.embed_size),
                initializer=tf.truncated_normal_initializer(mean=0, stddev=0.01))
            self.encoder_embeddings = tf.nn.embedding_lookup(
                params=self.embedding_matrix, ids=self.decoder_input_sequence, name='encoder_embeddings')
            self.decoder_embeddings = tf.nn.embedding_lookup(
                params=self.embedding_matrix, ids=self.decoder_output_sequence, name='decoder_embeddings')

    def compute_cost(self):
        crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=self.decoder_output_sequence, logits=self.output_sequence_logits)
        self.loss = tf.reduce_mean(crossent)

    def add_optimizer(self):
        '''
            clip gradient and apply gradient for fixing exploding graident problems.
        '''
        params = tf.trainable_variables()
        gradients = tf.gradients(self.loss, params)
        clipped_gradients, _ = tf.clip_by_global_norm(
            gradients, self.conf.max_gradient_norm)
        # No need for LR decay. As using adam, there is intrinsic lr decay.
        optimizer = tf.train.AdamOptimizer(self.conf.lr)
        self.train_op = optimizer.apply_gradients(
            zip(clipped_gradients, params))

    def forward_propagation_with_attention(self):
        with tf.variable_scope("encoder"):
            encoder_cell = rnn.BasicLSTMCell(
                num_units=self.conf.encoder_hidden_size)
            encoder_outputs, encoder_state = tf.nn.dynamic_rnn(encoder_cell,
                                                               self.encoder_embeddings,
                                                               time_major=False,
                                                               sequence_length=self.encoder_input_length,
                                                               dtype=tf.float32)

            attention_memory = tf.contrib.seq2seq.BahdanauAttention(self.conf.encoder_hidden_size,
                                                                    encoder_outputs)

        with tf.variable_scope("decoder"):
            decoder_cell = rnn.BasicLSTMCell(self.conf.decoder_hidden_size)
            decoder_cell = tf.contrib.seq2seq.AttentionWrapper(decoder_cell,
                                                               attention_memory,
                                                               attention_layer_size=self.conf.decoder_hidden_size)
            projection_layer = tf.layers.Dense(
                self.conf.vocab_size, use_bias=False)

        with tf.variable_scope("training"):
            helper = tf.contrib.seq2seq.TrainingHelper(self.decoder_embeddings, sequence_length=self.decoder_input_length,
                                                       time_major=False)
            decoder_inital_state = decoder_cell.zero_state(
                self.conf.batch_size, tf.float32).clone(cell_state=encoder_state)
            decoder = tf.contrib.seq2seq.BasicDecoder(decoder_cell,
                                                      helper,
                                                      decoder_inital_state,
                                                      output_layer=projection_layer)
            outputs_decoder, _, _ = tf.contrib.seq2seq.dynamic_decode(
                decoder)
            self.output_sequence_logits = outputs_decoder.rnn_output

        with tf.variable_scope("inference"):
            greedy_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
                self.embedding_matrix,
                tf.fill([self.conf.batch_size], self.vocab.encode(self.vocab.go)), self.vocab.encode(self.vocab.eos))
            # Decoder
            decoder_inital_state = decoder_cell.zero_state(
                self.conf.batch_size, tf.float32).clone(cell_state=encoder_state)
            decoder = tf.contrib.seq2seq.BasicDecoder(
                decoder_cell, greedy_helper, decoder_inital_state,
                output_layer=projection_layer)
            # setting maximum decoding length to encoder_input_length's 2 times during inference as it is unknown.
            maximum_iterations = tf.round(
                tf.reduce_max(self.encoder_input_length) * 2)
            # Dynamic decoding
            outputs_inference, _, _ = tf.contrib.seq2seq.dynamic_decode(
                decoder, maximum_iterations=maximum_iterations)
            self.predictions = outputs_inference.sample_id

    def run_batch(self, sess, encoder_input, decoder_input, decoder_output, is_training=False):
        encoder_input_length = np.full(
            len(encoder_input), len(encoder_input[0]))
        decoder_input_length = np.full(
            len(decoder_input), len(decoder_input[0]))
        fd = {self.encoder_input_sequence: encoder_input,
              self.decoder_input_sequence: decoder_input,
              self.decoder_output_sequence: decoder_output,
              self.encoder_input_length: encoder_input_length,
              self.decoder_input_length: decoder_input_length}
        ret = None
        if is_training:
            _, temp_cost = sess.run([self.train_op, self.loss], feed_dict=fd)
            ret = temp_cost
        else:
            pred, temp_cost = sess.run(
                [self.predictions, self.loss], feed_dict=fd)
            ret = (temp_cost, pred)
        return ret
