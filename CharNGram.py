# CharNGram.py
# Using Python 2.7.11

import itertools

"""
Splits sentence into character n-grams of length n

 @param sentence the training data
 @param n the n-gram length
 @return list of ngrams
"""
def getNGrams(sentence, n):
  sentence = (" " * (n - 1)) + sentence + " "
  return [sentence[i:i+n] for i in xrange(len(sentence) - n + 1)]


""" Creates the conditional frequency distribution

    @param sentences a list of sentences from the training data
    @param n the length of an n-gram
    @return a mapping of context (unique substrings of the first n-1 characters) to endings (last character) and their frequencies

  for each sentence in list of sentences:
        get n-grams using above method
        group by first n-1 characters (context)
        for each unique context, create a map of last character counts
          e.g., if you have the ngrams "chair" and "chain":
                the context would be "chai" and the last character counts would be "r -> 1" and "n -> 1"
        add the [ context -> last character counts ] mapping to your ultimate CFD (also a map)
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

class CharNGram:
  def __init__(self, language, conditionalCounts, n):
    self.language = language
    self.condCounts = conditionalCounts
    self.n = n
    self._getNormalizedCounts()

  def _getNormalizedCounts(self):
    for ctx, counts in self.condCounts.iteritems():
      charSum = 0.0
      for lastChar, count in counts.iteritems():
        charSum += count
      for lastChar, count in counts.iteritems():
        self.condCounts[ctx][lastChar] = (count + 1)/(charSum + 26)

    """ Using conditional frequency distribution, calculate and return p(c | ctx) """
  def ngramProb(self, ctx, c):
    if ctx in self.condCounts:
      if c in self.condCounts[ctx]:
        count = self.condCounts[ctx][c]
        if count > 0:
          return count
        else:
          return 1.0/(len(self.condCounts[ctx]) + 26)
      else:
        return 1.0/(len(self.condCounts[ctx]) + 26)
    else:
      return 1.0/26

    """ Multiply ngram probabilites for each ngram in word """
  def wordProb(self, word):
    prob = 1.0
    for ctx, counts in getConditionalCounts([word], self.n).iteritems():
      for lastChar, count in counts.iteritems():
        prob *= self.ngramProb(ctx, lastChar) * count
    return prob

