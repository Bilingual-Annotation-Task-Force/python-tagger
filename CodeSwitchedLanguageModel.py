# CodeSwitchedLanguageModel.py
# Using Python 2.7.11

import CharNGram

class codeSwitchedLanguageModel:
  def __init__(self, models):
    self.models = models

  def guess(self, word):
    highestProb = 0.0
    lowerWord = word.toLower()

    # Get maximum NGram probability over all models
    for model in self.models:
      modelProb = model.ngramProb(lowerWord)
      highestProb = max(highestProb, modelProb)

    # Find model with highest probability
    guess = []
    for model in self.models:
      if model.ngramProb(lowerWord) == highestProb:
        guess.append(model)

    # Functional calls in Python?
    # highestProb = max([model.ngramProb(word.toLower()) for model in models])
    # guess = filter(lambda x: models.x.ngramProb(word.toLower()) == highestProb)

    # Return language of model with highest probability
    return guess[0].language

  def prob(self, language, word):
    probList = []

    for model in self.models:
      if model.language == language:
        probList.append(model)

    prob = probList[0].ngramProb(word.toLower())

    # Functional calls in Python?
    # prob = filter(lambda x: models.x.language == language)[0].ngramProb(word.toLower())

    return prob

