'''
Unigram Language Model
'''

from .NGram import NGram
from os import linesep
from math import log10
from random import seed, random

class Unigram(NGram):
    '''
    Implements a unigram language model and builds a probability table based on
    the file passed to the constructor.
    '''

    def __init__(self, filename, laplace=False):
        super(Unigram, self).__init__(filename)
        self.name = "Unigram Language Model"
        if laplace:
            self.name += " (Laplace Smoothing)"
            for word in self.counter:
                self.counter[word] += 1
                self.total += 1

    def _entry_probability(self, entry):
        '''
        Return the unigram probility (cnt(word) / cnt(corpus))
        '''
        return float(self.counter[entry]) / self.total

    def _filecount(self, trainingfile):
        '''
        Count the occurances of individual words in a file
        '''
        for line in trainingfile.readlines():
            line = '<s> ' + line.rstrip() + ' </s>'

            for word in line.split():
                self.counter[word] += 1

        self.total = sum(self.counter.values())

    def compute_probability(self, string):
        '''
        Compute the probability of the supplied sentence based on the model.
        '''
        string = '<s> ' + string + ' </s>'

        probability = float()
        for word in string.split():
            try:
                probability += log10(float(self.counter[word]) / self.total)
            except ValueError:
                probability = float('-inf')

        return probability

    def generate_sentence(self, count):
        seed()

        dist = list()
        s = 0.0
        for word in [w for w in self.counter if w != '<s>']:
            p = self.counter[word] / (self.total - self.counter['<s>'])
            dist.append((word, p + s))
            s += p

        sentencelist = list()

        for i in range(0, count):
            sentence = '<s>'
            current_word = str()
            while current_word != '</s>':
                r = random()
                lastmass = 0.0
                for (word, mass) in dist:
                    if r > lastmass and r <= mass:
                        current_word = word
                        break
                    else:
                        lastmass = mass

                sentence += ' ' + current_word

            sentencelist.append(sentence)

        return sentencelist
