# CodeSwitchedLanguageModel.py
# Using Python 2.7.11

import CharNGram

class CodeSwitchedLanguageModel:
  def __init__(self, models):
    self.models = models

  def guess(self, word):
    highestProb = max(model.wordProb(word.lower()) for model in self.models)
    guess = [model for model in self.models
                   if model.wordProb(word.lower()) == highestProb]
    return guess[0].language

  def prob(self, language, word):
    return [model for model in self.models
                  if model.language == language][0].wordProb(word.lower())
