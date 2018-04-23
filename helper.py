class WordVoab():
    def __init__(self):
        self.word2ind = dict()
        self.ind2word = dict()
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
