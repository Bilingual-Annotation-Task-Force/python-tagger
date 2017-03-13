# CodeSwitchedLanguageModel.py
# Using Python 3.5

import cngram # Unused


class CodeSModel:
    """The code switched language model.
    This model effectively consists of a series of CNGrams.
    Args:
        models (CNGram):

    Properties:
        models (CNGram):
    """

    def __init__(self, models):
        self.models = models

    def guess(self, word):
        """Fetches the language a word is most likely to be in,
            based on the CodeSModel.
        Args:
            word (str): token to find the language of

        Return
            str: language the word is most likely to be in
        """
        # Efficiency consideration: Combine the following two steps into one
        #   loop so it is iterated through once instead of twice
        max_prob = max(model.word_prob(word.lower()) for model in self.models)
        guess = [model for model in self.models
                 if model.word_prob(word.lower()) == max_prob]
        # This should find the most common language... though I do agree that
        #   it would be incredibly rare for two to have the same probability
        return guess[0].lang

    def prob(self, lang, word):
        """Fetches the probability of a word to be in a language.

        Args:
            lang (str): The language to look through
            word (str): The word to scan the probabilities of

        Return:
            float: The probability of the word to be in a language
        """
        # Not sure how this works... mostly due to simply picking the first element
        return [model for model in self.models
                if model.lang == lang][0].word_prob(word.lower())
