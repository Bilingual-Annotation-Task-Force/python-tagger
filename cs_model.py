#  cs_model.py
#  Using Python 3.4.3

import cngram
from operator import itemgetter


class CodeSModel:
    """The code switched language model.
    This model effectively consists of a series of CNGrams.
    Args:
        models (CNGram):

    Properties:
        models (CNGram):
    """

    def __init__(self, models):
        self.models = {model.lang: model for model in models}

    def guess(self, word):
        """Fetches the language a word is most likely to be in,
            based on the CodeSModel.
        Args:
            word (str): token to find the language of

        Return
            str: language the word is most likely to be in
        """
        lower_word = word.lower()
        model_probs = ((lang, model.word_prob(lower_word)) for lang, model in
                       self.models.items())
        return max(model_probs, key=itemgetter(1))[0]

    def prob(self, lang, word):
        """Fetches the probability of a word to be in a language.

        Args:
            lang (str): The language to look through
            word (str): The word to scan the probabilities of

        Return:
            float: The probability of the word to be in a language
        """
        return self.models[lang].word_prob(word.lower())
