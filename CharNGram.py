# CharNGram.py
# Using Python 2.7.11

import itertools
import math

"""
Splits text into character n-grams of length n

 @param text the training data
 @param n the n-gram length
 @return list of ngrams
"""


def getNGrams(text, n):
  text = (" " * (n - 1)) + text + " "
  return [text[i:i+n] for i in xrange(len(text) - n + 1)]


""" Creates the conditional frequency distribution

    @param sentences a list of sentences from the training data
    @param n the length of an n-gram
    @return a mapping of context (unique substring of the first n-1 characters)
    to endings (last character) and their frequencies

  for each sentence in list of sentences:
        get n-grams using above method
        group by first n-1 characters (context)
        for each unique context, create a map of last character counts
          e.g., if you have the ngrams "chair" and "chain":
                the context would be "chai" and the last character counts
                would be "r -> 1" and "n -> 1"
        add the [ context -> last character counts ] mapping to your
        ultimate CFD (also a map)
          e.g., your CFD  would have lots of entries that look like:
                "chai" -> ["r" -> 1, "n -> 1]
                as an key-value pair.
"""


def getConditionalCounts(sentences, n):
  condCounts = {}
  for sentence in sentences:
    ngrams = getNGrams(sentence, n)
    for gram in ngrams:
      context, lastChar = gram[:n - 1], gram[-1]
      condCounts.setdefault(context, {}).setdefault(lastChar, 0)
      condCounts[context][lastChar] += 1
  return condCounts

"""
Revision of getConditionalCounts with Counters
def getConditionalCounts(sentences, n):
    ngrams = []
    for sentence in sentences:
      ngrams += getNGrams(sentences, n)
    contexts = [gram[:n - 1] for gram in ngrams]
    lastChars = [gram[-1] for gram in ngrams]
    counts = Counter(zip(contexts, lastChars))
    for (x, y), c in counts.iteritems():
      condCounts.setdefault(x, {})
      condCounts[x][y] = c
    return condCounts
"""


class CharNGram:
  def __init__(self, language, conditionalCounts, n, numLetters=26):
    self.language = language
    self.condCounts = conditionalCounts
    self.n = n
    self.numLetters = numLetters
    self._getNormalizedCounts()

  def _getNormalizedCounts(self):
    for ctx, counts in self.condCounts.iteritems():
      for lastChar, count in counts.iteritems():
        self.condCounts[ctx][lastChar] = (count + 1)/float(self.numLetters)

    """
    Using conditional frequency distribution,
    calculate and return p(c | ctx)
    """
  def ngramProb(self, ctx, c):
    return self.condCounts.get(ctx, {}).get(c, 1.0/float(self.numLetters))

    """ Multiply ngram probabilites for each ngram in word """
  def wordProb(self, word):
    prob = 1.0
    for ctx, counts in getConditionalCounts([word], self.n).iteritems():
      for lastChar, count in counts.iteritems():
        prob *= self.ngramProb(ctx, lastChar) * count
    return math.log(prob)
