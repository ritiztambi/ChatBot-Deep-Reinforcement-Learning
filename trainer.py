from neuralconversation import NeuralConversationModel
from helper import read_list_from_pickle, save_list_to_pickle, WordVoab

import tensorflow as tf


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

sample = read_list_from_pickle(".\\data\\ai_data.pkl")
vocab = read_list_from_pickle('.\\data\\vocab.pkl')
conf.vocab_size = len(vocab.word2ind)

model = NeuralConversationModel(conf, vocab)
model.add_place_holder()
model.add_embeddings()
model.forward_propagation_with_attention()
model.compute_cost()
model.add_optimizer()

with tf.Session() as sess:
    # model.run_batch(sess,sample[0][0][0])
    pass
    # code and test.
