import nltk
import codecs
import os

class WordVocab():
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
                    
                    aa = nltk.sent_tokenize(self.lines[q])
        print(aa)
       
            
            
   

def main():
    obj = WordVocab()
    obj.prepare_corpus()
     
        
if __name__ == "__main__":
    # calling main function
    main()

        
