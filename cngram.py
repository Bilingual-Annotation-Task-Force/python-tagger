# ngram.py
# Using Python 3.5

import itertools
import math


class CNGram:
    """Represents a section of text with n characters.

    Args:
        lang (str): The language of the n-gram
        cond_cnts ():
        num_letters (int): Number of letters in the original text.
        n (int, optional): The length of the n-gram. The default is 5.
    Properties:
        lang (str): n-gram language
        cond_cnt ():
        num_letters (int): Number of letters in the original text.
        n (int, optional): The length of the n-gram. The default is 5.
    """

    def __init__(self, lang, cond_cnts, num_letters, n = 5):
        self.lang = lang
        self.cond_cnts = cond_cnts
        self.num_letters = num_letters
        self.n = n
        self._normalize_counts()

    def _normalize_counts(self):
        """Normalizes the counts within the n-gram's cond_cnts"""
        for ctx, cnts in self.cond_cnts.iteritems():
            for lastc, cnt in cnts.iteritems():
                self.cond_cnts[ctx][lastc] = (cnt + 1)/float(self.num_letters)

    def ngram_prob(self, ctx, c):
        """Using conditional frequency distribution, calculate p(c | ctx).
        Return:
            float: P(c | ctx)
        """
        # Return the counts divided by the number of letters
        return self.cond_cnts.get(ctx, {}).get(c, 1.0/float(self.num_letters))

    def word_prob(self, word):
        """ Multiply n-gram probabilites for each n-gram in word
        Return:
            float: Probability of a certain word
        """
        prob = 1.0
        for ctx, cnts in get_cond_cnts([word], self.n).iteritems():
            for lastchar, cnt in cnts.iteritems():
                prob *= self.ngram_prob(ctx, lastchar) * cnt
        return math.log(prob)


def get_ngrams(text, n):
    """Splits text into character n-grams of length n.
    The provided string is padded with spaces at the beginning and end.

    Args:
        text(str): the training data
        n (int): the n-gram length
     Return:
        list<str>: n-gram strings
    """
    text = (" " * (n - 1)) + text + " "
    return [text[i:i+n] for i in range(len(text) - n + 1)]


def get_cond_cnts(sentences, n):
    """ Creates the conditional frequency distribution.
    The process is as follow:

    for each sentence,
        get n-grams using above method
        group by first n-1 characters (context)
        for each unique context, create a map of last character counts
            e.g., if you have the n-grams "chair" and "chain":
                the context would be "chai" and the last character counts
                would be "r -> 1" and "n -> 1"
         add the [ context -> last character counts ] mapping to your
         ultimate CFD (also a map)
             e.g., the CFD would have lots of entries that look like:
                 "chai" -> ["r" -> 1, "n -> 1]
                  as an key-value pair.

    Args:
        sentences (list<str>): sentences from the training data
        n (int): length of an n-gram
    Return:
        dict<str, dict<str, int>:  a mapping of context (unique substring of
        the first n-1 characters) to endings (last character) and their frequencies
    """
    cond_cnts = {}
    for sentence in sentences:
        ngrams = get_ngrams(sentence, n)
        for ngram in ngrams:
            ctx, lastc = ngram[:n - 1], ngram[-1]
            cond_cnts[ctx][lastc] = cond_cnts.setdefault(ctx, {}).setdefault(lastc, 0) + 1
    return cond_cnts
