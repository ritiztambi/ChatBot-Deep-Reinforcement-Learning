import nltk
import codecs
import os
import numpy as np
import random
from nltk.tokenize import TweetTokenizer
import operator
from collections import Counter
import pickle


class WordVoab():
    def __init__(self):
        self.word2ind = dict()
        self.ind2word = dict()
        self.word_freq = dict()

        self.unknown = '<unk>'
        self.eos = '<eos>'
        self.go = '<go>'
        self.pad = '<pad>'
        self.special_tokens = [self.unknown, self.go, self.eos, self.pad]

        for token in self.special_tokens:
            self.add_word(token)

    def add_word(self, word):
        if word not in self.word2ind:
            index = len(self.word2ind)
            self.word2ind[word] = index
            self.ind2word[index] = word
            self.word_freq[word] = 1
        else:
            self.word_freq[word] += 1

    def encode(self, word):
        if word not in self.word2ind:
            word = self.unknown
        return self.word2ind[word]

    def decode(self, ind):
        return self.ind2word[ind]

    def add_words(self, words):
        for word in words:
            self.add_word(word)

    def trim_dictionary(self, size):
        freq_dict = Counter(self.word_freq)
        # for 100 size chose most common 96(100-4 speical tokens) vocabulary.
        chosen_vocab = freq_dict.most_common(n=size - len(self.special_tokens))
        # rest old index mappings and word mappings
        self.word2ind = dict()
        self.ind2word = dict()
        self.word_freq = dict()
        # Now add the special tokens to the dictionary
        for word in self.special_tokens:
            self.add_word(word)
        # Now add most frequent words from chosen vocabulary list
        for word in chosen_vocab:
            self.add_word(word[0])
            self.word_freq[word[0]] = word[1]


class DatasetBuilder():
    def __init__(self, v):
        self.vocabulary = v

    def load_lines(self):
        self.lines = dict()
        file_name = os.getcwd() + "/data/movie_lines.txt"
        tknzr = TweetTokenizer(preserve_case=False)
        with codecs.open(file_name, "r", encoding='utf-8', errors='ignore') as fdata:
            lines = fdata.readlines()
            # read line by line
            for line in lines:
                values = line.split(" +++$+++ ")
                line_id = str(values[0])
                text = values[-1]
                # tokenize text and add to vocabulary
                tokens = tknzr.tokenize(text)
                self.vocabulary.add_words(tokens)
                self.lines[line_id] = tokens
                #print(line_id, tokens)
            fdata.close()

    def load_conversation(self):
        self.conversations = []
        file_name = os.getcwd() + "/data/movie_conversations.txt"
        with codecs.open(file_name, "r", encoding='utf-8', errors='ignore') as fdata:
            lines = fdata.readlines()
            for line in lines:
                values = line.strip().split(" +++$+++ ")
                line_ids = list(map(str.strip, values[-1][1:-1].split(",")))
                self.conversations.append(line_ids)
            fdata.close()

    def prepare_corpus(self, size=None):
        '''
            Pass -1, if you don't want to trim vocab.
            Reads lines and convesations and builds the needed [(Q,A)] list.
        '''
        step = 2
        self.load_lines()
        if size != None:
            self.vocabulary.trim_dictionary(size)
        self.load_conversation()
        self.corpus = []
        for conv in self.conversations:
            if len(conv) >= step:
                for i in range(0, len(conv) - step + 1):
                    q = conv[i].strip()[1:-1]
                    a = conv[i + 1].strip()[1:-1]
                    # split, strip and store as [(Q,A)] pairs.
                    q_line = list(map(self.vocabulary.encode, self.lines[q]))
                    a_line = list(map(self.vocabulary.encode, self.lines[a]))
                    self.corpus.append([q_line, a_line])


class BatchBuilder():
    def __init__(self, v):
        if not isinstance(v, WordVoab):
            raise Exception('Please initialize with a vocabulary!!')
        self.vocabulary = v

    def pad_batch(self, batch):
        maximumEncLen = -1
        maximumDecLen = -1
        # find the maximum encoder and decoder length in current input batch and use it later to pad the input and make it to same size.
        for b in range(len(batch)):
            maximumEncLen = max(maximumEncLen, len(batch[b][0]))
            maximumDecLen = max(maximumDecLen, len(batch[b][1]))

        maximumDecLen = min(maximumDecLen, 20)
        maximumEncLen = min(maximumEncLen, 20)

        v = self.vocabulary
        fixed_batch = []

        # iterate through the batch and append needed paddings.
        for b in range(len(batch)):
            needed_encoder_paddings = max(0, maximumEncLen - len(batch[b][0]))
            needed_decoder_paddings = max(0, maximumDecLen - len(batch[b][1]))

            if len(batch[b][0]) > 20:
                batch[b][0] = batch[b][0][:20]

            if len(batch[b][1]) > 20:
                batch[b][1] = batch[b][1][:20]

            encoder_ips = [v.encode(v.pad)] * \
                needed_encoder_paddings + batch[b][0]
            decoder_ips = [v.encode(v.go)] + batch[b][1] + \
                [v.encode(v.pad)] * needed_decoder_paddings
            decoder_ops = batch[b][1] + \
                [v.encode(v.eos)] + [v.encode(v.pad)] * needed_decoder_paddings
            fixed_batch.append([encoder_ips, decoder_ips, decoder_ops])
        return fixed_batch

    def generate_batches(self, training_samples, batch_size=3):
        random.shuffle(training_samples)
        start_index = 0
        batches = []
        while start_index < len(training_samples):
            batch = training_samples[start_index:min(
                start_index + batch_size, len(training_samples))]
            batches.append(self.pad_batch(batch))
            start_index += batch_size
        return batches


def save_list_to_pickle(listData, fileName):
    with open(fileName, 'wb') as f:
        pickle.dump(listData, f)


def read_list_from_pickle(fileName):
    with open(fileName, 'rb') as f:
        mynewlist = pickle.load(f)
    return mynewlist


if __name__ == '__main__':
    v = WordVoab()
    data_builder = DatasetBuilder(v)
    data_builder.prepare_corpus(size=10000)
    print(len(v.word2ind), len(v.ind2word), len(v.word_freq))

    batch_builder = BatchBuilder(v)
    training_samples = batch_builder.generate_batches(
        data_builder.corpus, batch_size=64)

    print(len(training_samples))

    #save_list_to_pickle(data_builder.corpus, '.\\data\\ai_data.pkl')
    #save_list_to_pickle(v, '.\\data\\vocab.pkl')
    #save_list_to_pickle(training_samples, '.\\data\\ai_training_data.pkl')
