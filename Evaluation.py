#  Evaluation.py
#  Using Python 2.7.11

import sys
import re
import io
import HiddenMarkovModel
import string
from nltk.tag.stanford import StanfordNERTagger


""" Splits text input into words and formats them, splitting by whitespace

    @param text a string of text
    @return a list of formatted words
"""
# case-insensitive tokenizer for ngram probabilities only


def toWords(text):
    # requires utf-8 encoding
    token = re.compile(ur'[\w]+|[^\s\w]', re.UNICODE)
    tokens = re.findall(token, text)
    return [word.lower() for word in tokens]


class Evaluator:
    def __init__(self, cslm, hmm):
        self.cslm = cslm
        self.hmm = hmm
        self.engClassifier = StanfordNERTagger(
            "../stanford-ner-2015-04-20/classifiers/english.all.3class.distsim.crf.ser.gz",
            "../stanford-ner-2015-04-20/stanford-ner.jar")
        self.spanClassifier = StanfordNERTagger(
            "../stanford-ner-2015-04-20/classifiers/spanish.ancora.distsim.s512.crf.ser.gz",
            "../stanford-ner-2015-04-20/stanford-ner.jar")
        self._tagger()

    def _tagger(self):
        # can hmm accept accept list?
        hmmtags = self.hmm.generateTags()
        words = self.hmm.words  # this needs to be case-sensitive
        taggedTokens = [("Token", "Language", "Named Entity")]

        for k, word in enumerate(words):

            # check if punctuation else use hmmtag
            lang = 'Punct' if word in string.punctuation else hmmtags[k]

            # check if word is NE
            try:
                engTag = self.engClassifier.tag(words[k-2:k+2])[2][1]
                spanTag = self.spanClassifier.tag(words[k-2:k+2])[2][1]

            except IndexError:
                engTag = self.engClassifier.tag([word])[0][1]
                spanTag = self.spanClassifier.tag([word])[0][1]

            # mark as NE either classifier identifies it
            if engTag != 'O' or spanTag != 'O':
                NE = "{}/{}".format(engTag, spanTag)
            else:
                NE = "O"
            taggedTokens.append((word, lang, NE))
        return taggedTokens

    #  Write annotation to output file
    def annotate(self, textfile):
        with io.open(textfile + '_annotated.txt', 'w', encoding='utf-8') as output:
            text = io.open(textfile).read()
            for line in self._tagger(text):
                print>>output, "{},{},{}".format(*line)

    #  Write evaluation of annotation to file
    def evaluate(self, goldStandard):
        with io.open(goldStandard + '_outputwithHMM.txt', 'w', encoding='utf8') as output:
            lines = io.open(goldStandard, 'r', encoding='utf8').readlines()
            text = [x.split(",")[1] for x in lines]
            gold_tags = [x.split(",")[2] for x in lines]
            annoated_output = self._tagger(text)
            lang_tags = [x.split(",")[1] for x in annoated_output]
            ne_tags = [x.split(",")[2] for x in annoated_output]
            langCorrect = langTotal = NECorrect = NETotal = 0
            evaluations = []

            # compare gold standard and model tags
            for word, gold, lang, NE in zip(text, gold_tags, lang_tags, ne_tags):

                # evaluate language tags
                if gold in ('Eng', 'Spn'):
                    langTotal += 1
                    if gold == 'Eng' and lang == 'Eng' and NE==0:
                        langCorrect += 1
                        evaluations.append("Correct")
                    elif gold == 'Spn' and lang == 'Spn'and NE==0:
                        langCorrect += 1
                        evaluations.append("Correct")
                    else:
                        evaluations.append("Incorrect")
                # evaluate NE tags
                elif gold == "NE":
                    NETotal += 1
                    if NE != 'O':
                        NECorrect += 1
                        evaluations.append("Correct")
                    else:
                        evaluations.append("Incorrect")
                # don't evaluate punctuation
                else:
                    evaluations.append("NA")

            print>>output, "Language Accuaracy: {}".format(langCorrect / float(langTotal))
            print>>output, "NE Accuaracy: {}".format(NECorrect / float(NETotal))
            for word, gold, lang, NE in zip(text, gold_tags, lang_tags, ne_tags, evaluations):
                print>>output, "{},{},{},{},{}".format()

"""
Process arguments
Get corpora and create NGram models
Create Code-Switch Language Model
Build Markov model with Expectation Minimization
Annotate
Evaluate
"""


def main(argv=sys.argv):
    testCorpus = 'INSERT RELATIVE PATH HERE'  # Extract from arguments?
    goldStandard = 'INSERT RELATIVE PATH HERE'
    n = 5
    engData = toWords(io.open('PATH TO ENG DATA', 'r', encoding='utf8').read())
    spanData = toWords(io.open('PATH TO SPAN DATA', 'r', encoding='utf8').read())
    enModel = NGramModel('Eng', getConditionalCounts(engData, n), n)
    esModel = NGramModel('Spn', getConditionalCounts(spanData, n), n)

    cslm = CodeSwitchedLanguageModel([enModel, esModel])

    testWords = 0

    tags = ['Eng', 'Spn']

    transitions = 0  # Insert Expectation Maximization call here
    hmm = HiddenMarkovModel(testWords, tags, transitions, cslm)

    eval = Evaluator(cslm, hmm)
    eval.annotate(testCorpus)

    #  Use an array of arguments?
    #  Should user pass in number of characters, number of languages, names of
    #  languages?

if __name__ == "__main__":
    main()
