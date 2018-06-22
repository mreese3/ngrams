'''
NGram Language Model
'''
from collections import Counter
from math import log10
from os import linesep
from itertools import product
import re
import random

class NGram(object):
    '''
    Implements an n-gram language model.  Arguments:
        -N: the size of the word tuple
        -trainingdata: the file-like object to read the training data line by line
        -smoothing: If True, use Laplace Add-1 smoothing for the model
    '''
    def __init__(self, N, trainingdata, smoothing=False):
        self.ngrams = Counter()
        self.wordcount = Counter()
        self.N = N
        self.regex = None
        self.smoothing = smoothing
        self._read_training_data(trainingdata)

##############################################################################
############################# Private Methods ################################
##############################################################################

    def _read_training_data(self, trainingdata):
        '''
        Reads training data from the file-like object trainingdata.
        For internal use only.  Used by constructor to read the file passed.
        '''

        # build a regular expression for line parsing (based on N)
        # The regex should end up something like this:
        # "([\w<>/]+) (?=([\w<>/]+) ... ([\w<>/]+))"
        # Where the amount of look-ahead groups is equal to N - 1
        regexstring = "([\w<>/]+)"
        if self.N > 1:
            regexstring += " (?=([\w<>/]+)"
            for i in range(0, self.N - 2):
                regexstring += " ([\w<>/]+)"
            regexstring += ")"
        self.regex = re.compile(regexstring)

        for line in trainingdata.readlines():
            # Add the start and stop symbol the line
            line = '<s> ' + line.rstrip() + ' </s>'

            # If N > 1, then the regex generated above will never match the
            # last N-1 words in the line, so they need to be added to the
            # wordcount manually
            for i in range(1, self.N):
                self.wordcount[line.split()[-i]] += 1

            for ngram in re.findall(self.regex, line):
                if self.N == 1:
                    self.wordcount[ngram] += 1
                    # Note: items in the ngrams counters are always tuples,
                    # even if it is a single word (this makes processing
                    # consistent in other parts of the code)
                    self.ngrams[(ngram,)] += 1
                else:
                    self.wordcount[ngram[0]] += 1
                    self.ngrams[ngram] += 1

        # If we are smoothing the model, then if N is 1, just add one to every
        # word in the wordcounter and ngram counter. If N is greater than 1,
        # get the Cartesian Product of the wordcount set by itself and add all
        # of the ngrams generated to the ngram counter.
        if self.smoothing:
            if self.N == 1:
                for word in self.wordcount:
                    self.wordcount[word] += 1
                    self.ngrams[(word,)] += 1
            else:
                allngrams = product(self.wordcount, repeat=self.N)
                for ngram in allngrams:
                    self.ngrams[ngram] += 1


    def _pmf(self, cmpfunc):
        '''
        Build a probability mass function table based upon the condition
        specified in the lambda cmpfunc.
        For internal use only.  Used by the sentence generator.
        '''
        mass = list()
        psum = 0.0
        for ngram in [n for n in self.ngrams if cmpfunc(n)]:
            probability = self.ngram_probability(ngram)
            mass.append((ngram, probability + psum))
            psum += probability

        return mass


    def _pmf_selection(self, r, mass_distribution):
        '''
        Loop through the proabability masses in the distribution, comparing r
        to the current_mass.  If r is within the current mass, return the ngram
        associated with said mass.
        For internal use only.  Used by the sentence generator.
        '''
        lastmass = 0.0
        return_ngram = None
        for (ngram, mass) in mass_distribution:
            if r > lastmass and r <= mass:
                return_ngram = ngram
                break
            else:
                lassmass = mass
        return return_ngram

##############################################################################
############################## Public Methods ################################
##############################################################################

    def ngram_probability(self, ngram):
        '''
        Returns the individual probability of the ngram passed.
            -ngram: Some iterable of strings
        '''
        probability = 0.0
        if self.N == 1:
            probability = self.ngrams[ngram] / sum(self.wordcount.values())
        else:
            # This may be wrong for N > 2; Verify
            probability = self.ngrams[ngram] / sum([value for (key, value) in
                self.ngrams.items() if key[:-1] == ngram[:-1]])
        return probability


    def sentence_probability(self, sentence):
        '''
        Returns the probability of a given sentence in log10.
            -sentence: string of words in the model vocabulary
        '''
        sentence = '<s> ' + sentence + ' </s>'

        probability = 0.0

        for ngram in re.findall(self.regex, sentence):
            if self.N == 1:
                ngram = (ngram,)
            try:
                probability += log10(self.ngram_probability(ngram))
            except ValueError:
                probability = float('-inf')

        return probability


    def get_model_formatted(self):
        '''
        Returns a string of the ngram model formatted for printing.
        '''
        returnstring = None
        if self.N == 1:
            returnstring = "Unigram Model"
        elif self.N == 2:
            returnstring = "Bigram Model"
        elif self.N == 3:
            returnstring = "Trigram Model"
        else:
            returnstring = "%d-gram Model" % self.N

        # Create a copy of the ngram set to do the actual printing
        # This is so if we are doing smoothing, we only print the original
        # ngrams present after smoothing (those with a count > 1)
        working_copy = self.ngrams
        if self.smoothing:
            returnstring += " (Laplace Add-1 Smoothing)"
            # remove all of the 1-counted ngrams from the copy
            working_copy = Counter({key: value for (key, value) in
                self.ngrams.items()
                if value > 1})

        # Generate our table header
        returnstring += ":" + linesep
        returnstring += "N-Gram".ljust(self.N * 10)
        returnstring += "Count".ljust(8)
        returnstring += "Probability"

        for ngram in working_copy:
            returnstring += linesep
            if self.N == 1:
                returnstring += str(ngram[0]).ljust(10*self.N)
            else:
                ngram_string='[' + ngram[0]
                for i in range(1, len(ngram)):
                    ngram_string += ', ' + ngram[i]
                ngram_string += ']'
                returnstring += ngram_string.ljust(10*self.N)
            returnstring += str(working_copy[ngram]).ljust(8)
            returnstring += "%0.4f" % self.ngram_probability(ngram)

        return returnstring


    def generate_sentence(self, sentence_count=1):
        '''
        Generate a sentence using a biased random selection process based on
        our model.  Multiple sentences can be generated by passing
        sentence_count.  Returns the sentence (or sentences) as a list of
        strings.
        '''
        random.seed()

        sentences = list()

        for i in range(0, sentence_count):
            current_sentence = str()
            current_ngram = None
            mass_distribution = list()
            if self.N > 1:
                # if N is greater than 1, then the initial mass distribution
                # will need to be for all of the ngrams that start with '<s>'
                # Then, an ngram is chosen and the entire ngram is added to the
                # sentence as the start.
                mass_distribution = self._pmf(lambda x : x[0] == '<s>')
                r = random.random()
                current_ngram = self._pmf_selection(r, mass_distribution)
                current_sentence = " ".join(current_ngram)
            else:
                # if N is 1, then the mass distribution is the word
                # probabilities.
                mass_distribution = self._pmf(lambda x : True)
                current_ngram = ('<s>',)
                current_sentence = '<s>'

            # Loop until our latest word is the stop symbol
            while current_ngram[-1] != '</s>':
                r = random.random()
                if self.N > 1:
                    # If N is greater than 1, the mass distribution needs to be
                    # regenerated every iteration to reflect what our new
                    # current word is.
                    # Note: The lambda used here works for Unigrams and
                    # Bigrams, but it has not been tested for Trigrams or other
                    # N-Grams.
                    mass_distribution = self._pmf(lambda x : x[:-1] ==
                            current_ngram[1:])

                current_ngram = self._pmf_selection(r, mass_distribution)
                current_sentence += " " + current_ngram[-1]
            sentences.append(current_sentence)
        return sentences
