# CharNGram.py

"""
Splits string into character n-grams of length n
 @param string the training data
 @param n the n-gram length
"""
def getNGrams(sentence, n):
  sentence = (" " * (n - 1)) + sentence + " "   # Append start and end spaces
  return [string[i:i+n] for i in range(len(sentence) - n + 1)]


""" Creates the conditional frequency distribution

    @param sentences a list of sentences from the training data
    @param n the length of an n-gram
    @return a mapping of context (unique substrings of the first n-1 characters) to endings (last character) and their frequencies
"""
def getConditionalCounts(sentences, n):
  """ for each sentence in list of sentences:
        get n-grams using above method
        group by first n-1 characters (context)
        for each unique context, create a map of last character counts
          e.g., if you have the ngrams "chair" and "chain":
                the context would be "chai" and the last character counts would be "r -> 1" and "n -> 1"
        add the [ context -> last character counts ] mapping to your ultimate CFD (also a map)
          e.g., your CFD  would have lots of entries that look like: 
                "chai" -> ["r" -> 1, "n -> 1]
                as an key-value pair.
        

      (NOTE: you don't have to structure your conditional frequency distribution in this particular way; I just found it more intuitive.
      If it doesn't make sense I can go over it on gchat, or feel free to come up with your own implementation if another way makes more sense!)
"""

class CharNGram:
  def __init__(self, name, conditionalCounts, n):
    self.name = name
    self.data = data
    self.n = n


  """ Using conditional frequency distribution, calculate and return p(c | ctx) """
  def ngramProb(ctx, c):
    return c / ctx
  
  """ Multiply ngram probabilites for each ngram in word """
  def wordProb(word):
    return 0.0

