# CharNGram.py

"""
Splits string into character n-grams of length n

 @param string the training data
 @param n the n-gram length
 @return list of ngrams
"""
def getNGrams(sentence, n):
  sentence = (" " * (n - 1)) + sentence + " "
  return [string[i:i+n] for i in range(len(sentence) - n + 1)]


""" Creates the conditional frequency distribution

    @param sentences a list of sentences from the training data
    @param n the length of an n-gram
    @return a mapping of context (unique substrings of the first n-1 characters) to endings (last character) and their frequencies
      a dictionary of mapping between the first n - 1 characters of an ngram and
      a list of tuples of the ending characters

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
  freqCount = {}
  for sentence in sentences:
    ngrams = getNGrams(sentence, n)
    for gram in ngrams:
      context = gram[:n - 1]
      if not context in freqCount:
        freqCount[context] = [(gram[-1], 1)]
      else:
        freqCount[context].append((gram[-1], 1))
  return freqCount


class CharNGram:
  def __init__(self, name, conditionalCounts, n):
    self.name = name
    self.data = data
    self.n = n


  """ Using conditional frequency distribution, calculate and return p(c | ctx) """
  def ngramProb(ctx, c):
    return 0.0
  
  """ Multiply ngram probabilites for each ngram in word """
  def wordProb(word):
    return 0.0

