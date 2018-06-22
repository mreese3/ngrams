'''
Base Class for Unigram and Bigram models
'''

from collections import Counter
from sys import exit
from os import linesep


class NGram(object):
    '''
    Use as a base class for both the unigram and bigram models.
    Not meant to be instantiated itself, so most methods raise a
    NotImplementedError when called.
    '''

    def __init__(self, filename):
        self.counter = Counter()
        self.total = 0
        self.name = None

        # Try to open the file and call the implementation-specific counting
        # function
        try:
            trainingfile = open(filename)
            self._filecount(trainingfile)

        except IOError as e:
            print("Could Not Open File: " + e.strerror)
            exit(1)

    def _filecount(self, trainingfile):
        '''
        Implementation specific line processing.
        To be implemented in inheriting classes.
        '''
        raise NotImplementedError("Not Implemented in Base Class")

    def _entry_probability(self, entry):
        '''
        Returns the probability of an individual entry
        To be implemented in inheriting classes.
        '''
        raise NotImplementedError("Not Implemented in Base Class")

    def get_model_info_formatted(self):
        '''
        Returns a string with model info in a pretty format
        '''
        returnstring = self.name + ':'

        for entry in self.counter:
            returnstring += linesep
            returnstring += str(entry).ljust(20)
            returnstring += str(self.counter[entry]).ljust(6)
            returnstring += "%0.4f" % self._entry_probability(entry)

        return returnstring

    def compute_probability(self, string):
        '''
        Computes the probability of the sentence passed as string according to
        the model.
        To be implemented by the inheriting classes
        '''
        raise NotImplementedError("Not Implemented in Base Class")

    def generate_sentence(self, count):
        '''
        Generate N sentences according to the model where N = count
        To be implemented by the inheriting classes
        '''
        raise NotImplementedError("Not Implemented in Base Class")
