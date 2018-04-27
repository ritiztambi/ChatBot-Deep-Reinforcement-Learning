import nltk
import codecs
import os
import numpy as np
import random


class WordVoab():
    def __init__(self):
        self.word2ind = dict()
        self.ind2word = dict()
        self.lines = dict()
        self.conversations = []
        self.unknown = '<unk>'
        self.eos = '<eos>'
        self.go = '<go>'
        self.pad = '<pad>'
        self.word_freq = dict()
        self.add_word(self.unknown)
        self.add_word(self.eos)
        self.add_word(self.go)
        self.add_word(self.pad)
        self.size = 10000
        self.corpus=[]
        

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
    
    def load_lines(self):
        file_name=os.getcwd()+"/data/movie_lines.txt"
        with codecs.open(file_name, "r",encoding='utf-8', errors='ignore') as fdata:
            lines = fdata.readlines()
            for line in lines:
                values=line.split(" +++$+++ ")
                line_id= str(values[0])
                text= values[-1]
                self.lines[line_id] = text
        fdata.close()
    
    def load_conversation(self):
        file_name=os.getcwd()+"/data/movie_conversations.txt"
        with codecs.open(file_name, "r",encoding='utf-8', errors='ignore') as fdata:
            lines=fdata.readlines()
            for line in lines:
                values=line.split(" +++$+++ ")
                line_ids=values[-1][1:-2].strip().split(",")
                self.conversations.append(line_ids)
        fdata.close()
        
    def prepare_corpus(self):
        step=2
        self.load_lines()
        self.load_conversation()
        for conv in self.conversations:
            if len(conv) >=step:
                for i in range(0,len(conv)-step+1):
                    q=conv[i].strip()[1:-1]
                    a=conv[i+1].strip()[1:-1]
                    self.corpus.append([self.lines[q],self.lines[a]])
       
            


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

        v = self.vocabulary
        fixed_batch = []

        # iterate through the batch and append needed paddings.
        for b in range(len(batch)):
            needed_encoder_paddings = maximumEncLen - len(batch[b][0])
            needed_decoder_paddings = maximumDecLen - len(batch[b][1])
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

'''
if __name__ == '__main__':
    training_samples = []
    for _ in range(10):
        Q = list(np.random.randint(0, 5, size=np.random.randint(3, 6)))
        A = list(np.random.randint(0, 5, size=np.random.randint(3, 6)))
        training_samples.append([Q, A])
    batcher = BatchBuilder(WordVoab())
    bs = batcher.generate_batches(training_samples)
    for b in bs:
        for b_ in b:
            print(b_)
        print('***')
'''

