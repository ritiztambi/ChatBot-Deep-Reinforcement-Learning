from neuralconversation import NeuralConversationModel
from helper import read_list_from_pickle, save_list_to_pickle, WordVoab

import tensorflow as tf
import numpy as np


class Config():
    def __init__(self):
        self.batch_size = 64
        self.encoder_sequence_length = 10
        self.decoder_sequence_length = 5
        self.lr = 0.01
        self.encoder_hidden_size = 100
        self.decoder_hidden_size = 100
        self.keep_prob = 1.0
        self.num_epochs = 5
        self.embed_size = 50
        self.vocab_size = -1
        self.max_gradient_norm = 5
        self.stride = 2


conf = Config()

sample = read_list_from_pickle(".\\data\\ai_training_data.pkl")
vocab = read_list_from_pickle('.\\data\\vocab.pkl')
conf.vocab_size = len(vocab.word2ind)

tf.reset_default_graph()

model = NeuralConversationModel(conf, vocab)
model.add_place_holder()
model.add_embeddings()
model.forward_propagation_with_attention()
model.compute_cost()
model.add_optimizer()

init = tf.global_variables_initializer()


def transpose_batch(batch):
    enocder_ips = []
    decoder_ips = []
    decoder_ops = []

    for datapoint in batch:
        enocder_ips.append(datapoint[0])
        decoder_ips.append(datapoint[1])
        decoder_ops.append(datapoint[2])

    return np.array(enocder_ips), np.array(decoder_ips), np.array(decoder_ops)


def run_epoch(sess, sample):
    losses = []
    for batch in sample:
        encoder_input, decoder_input, decoder_output = transpose_batch(batch)
        loss = model.run_batch(sess, encoder_input, decoder_input,
                               decoder_output, is_training=True)
        losses.append(loss)
    return np.mean(losses)


with tf.Session() as sess:
    sess.run(init)
    loss = run_epoch(sess, sample)
    print(loss)
