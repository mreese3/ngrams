'''
Bigram Language Model
'''

from .NGram import NGram
from collections import Counter
from math import log10
from re import findall
from random import random, seed

class Bigram(NGram):
    '''
    Implements a bigram language model and builds a probability table based on
    the file passed.
    '''

    def __init__(self, filename, laplace=False):
        self.unigram_counter = Counter()
        super(Bigram, self).__init__(filename)
        self.name = "Bigram Language Model"
        self.laplace = False
        if laplace:
            self.laplace = True
            self.name += " (Laplace Smoothing)"
            for word1 in self.unigram_counter:
                for word2 in self.unigram_counter:
                    self.unigram_counter[word1] += 1
                    self.counter[(word1, word2)] += 1

    def _entry_probability(self, entry):
        '''
        Returns the bigram probability
        '''
        return float(self.counter[entry]) / self.unigram_counter[entry[0]]

    def _filecount(self, trainingfile):
        for line in trainingfile.readlines():
            line = '<s> ' + line.rstrip() + ' </s>'

            # The regex match below will never match the last word in a string,
            # so add to the </s> counter here
            self.unigram_counter['</s>'] += 1

            for bigram in findall('([\w<>/]+) (?=([\w<>/]+))', line):
                self.counter[bigram] += 1
                self.unigram_counter[bigram[0]] += 1

        self.total = sum(self.counter.values())

    def get_model_info_formatted(self):
        '''
        If we are doing laplace smoothing, remove all the 1 values and call the
        string function.  If not, call the string function like normal
        '''
        if self.laplace:
            tmp = self.counter.copy()
            self.counter = Counter({pair: tmp[pair] for pair in tmp if tmp[pair] > 1})
            returnstring = super(Bigram, self).get_model_info_formatted()
            self.counter = tmp.copy()
            return returnstring
        else:
            return super(Bigram, self).get_model_info_formatted()

    def compute_probability(self, string):
        '''
        Compute the probability of the supplied sentence based on the model.
        '''
        string = '<s> ' + string + ' </s>'

        probability = float()

        for pair in findall(r'([\w<>/]+) (?=([\w<>/]+))', string):
            try:
                probability += log10(float(self.counter[pair]) /
                        self.unigram_counter[pair[0]])
            except ValueError:
                probability = float('-inf')

        return probability

    def generate_sentence(self, count):
        seed()

        sentence_list = list()
        massdist = list()


        for i in range(0, count):
            sentence = '<s>'
            current_pair = ('', '<s>')
            while current_pair[1] != '</s>':

                massdist.clear()
                # generate our mass distribution function for the current
                # existing word
                s = 0.0
                for pair in [p for p in self.counter if p[0] == current_pair[1]
                        and p[1] != '<s>']:
                    p = self._entry_probability(pair)
                    massdist.append((pair, p + s))
                    s += p

                r = random()
                lastmass = 0.0
                for (pair, mass) in massdist:
                    if r > lastmass and r <= mass:
                        current_pair = pair
                        break
                    else:
                        lastmass = mass

                sentence += ' ' + current_pair[1]

            sentence_list.append(sentence)

        return sentence_list
