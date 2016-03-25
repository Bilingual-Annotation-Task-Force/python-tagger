# CodeSwitchedLanguageModel.py
# Using Python 2.7.11

import CharNGram

class codeSwitchedLanguageModel:
  def __init__(self, models):
    self.models = models

  def guess(self, word):
    highestProb = max(map(lambda x: x.wordProb(word.lower()), self.models))
    guess = filter(lambda x: x.wordProb(word.lower()) == highestProb, self.models)
    return guess[0].language

  def prob(self, language, word):
    return filter(lambda x: x.language == language, self.models)[0].wordProb(word.lower())
