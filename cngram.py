#  cngram.py
#  Using Python 3.4.3

import math
from collections import defaultdict, Counter


class CNGram:
    """Represents a section of text with n characters.

    Args:
        lang (str): The language of the n-gram model
        words (list<str>): Tokenized words for a single language.
        num_letters (int): Number of letters in the original text. The default
            is 26.
        n (int, optional): The length of the n-gram. The default is 5.
    Properties:
        lang (str): n-gram language
        words (list<str>): Tokenized words for a single language.
        num_letters (int): Number of letters in the original text. The default
            is 26.
        n (int, optional): The length of the n-gram. The default is 5.
    """

    def __init__(self, lang, words, num_letters=26, n=5):
        self.lang = lang
        self.cond_cnts = get_cond_cnts(words, n)
        self.num_letters = num_letters
        self.n = n
        self._normalize_counts()

    def _normalize_counts(self):
        """Normalizes the counts within the n-gram's cond_cnts"""
        for ctx, cnts in self.cond_cnts.items():
            for lastc, cnt in cnts.items():
                ctx_size = len(self.cond_cnts[ctx])
                self.cond_cnts[ctx][lastc] = (cnt + 1)/float(ctx_size + self.num_letters)

    def ngram_prob(self, ctx, c):
        """Using conditional frequency distribution, calculate p(c | ctx).
        Return:
            float: P(c | ctx)
        """
        # Return the counts divided by the number of letters
        ctx_size = len(self.cond_cnts[ctx])
        return self.cond_cnts.get(ctx, {}).get(c, 1.0/float(ctx_size + self.num_letters))

    def word_prob(self, word):
        """ Multiply n-gram probabilites for each n-gram in word
        Return:
            float: Probability of a certain word
        """
        prob = 1.0
        for ctx, cnts in get_cond_cnts([word], self.n).items():
            for lastchar, cnt in cnts.items():
                prob *= self.ngram_prob(ctx, lastchar) * cnt
        return math.log(prob)


def get_ngrams(word, n):
    """Splits word into character n-grams of length n.
    The provided string is padded with spaces at the beginning and end.

    Args:
        word(str): the training data
        n (int): the n-gram length
     Return:
        list<str>: n-gram strings
    """
    pad = " " * (n - 1)
    word = pad + word + pad
    return (word[i:i+n] for i in range(len(word) - n + 1))


def get_cond_cnts(words, n):
    """ Creates the conditional frequency distribution.
    The process is as follows:

    for each word,
        get n-grams using above method
        group by first n-1 characters (context)
        for each unique context, create a map of last character counts
            e.g., if you have the n-grams "chair" and "chain":
                the context would be "chai" and the last character counts
                would be "r -> 1" and "n -> 1"
         add the [ context -> last character counts ] mapping to your
         ultimate CFD (also a map)
             e.g., the CFD would have lots of entries that look like:
                 "chai" -> ["r" -> 1, "n" -> 1]
                  as an key-value pair.

    Args:
        words (list<str>): words from the training data
        n (int): length of an n-gram
    Return:
        dict<str, dict<str, int>:  a mapping of context (unique substring of
        the first n-1 characters) to endings (last character) and their
        frequencies
    """
    cond_cnts = defaultdict(Counter)
    for words in words:
        ngrams = get_ngrams(words, n)
        for ngram in ngrams:
            ctx, lastc = ngram[:n - 1], ngram[-1]
            cond_cnts[ctx][lastc] += 1
    return cond_cnts
