#  Evaluation.py
#  Using Python 2.7.11
#trial line August 11, 2016
import sys
import re
import io
from HiddenMarkovModel import HiddenMarkovModel
import string
from nltk.tag.stanford import StanfordNERTagger
from collections import Counter
from CharNGram import *
from CodeSwitchedLanguageModel import CodeSwitchedLanguageModel
import math
import csv

""" Splits text input into words and formats them, splitting by whitespace

    @param text a string of text
    @return a list of formatted words
"""
# case-insensitive tokenizer for ngram probabilities only


def toWords(text): # separates punctuation
    # requires utf-8 encoding
    token = re.compile(ur'[\w]+|[^\s\w]', re.UNICODE)
    tokens = re.findall(token, text)
    return [word.lower() for word in tokens]
"""

def toWords(text): # splits on white space
    tokens = re.sub("\t|\n|\r", "", text)
    return [word.lower() for word in tokens.split()]
"""

def toWordsCaseSen(text): # separates punctuation
    # requires utf-8 encoding
    token = re.compile(ur'[\w]+|[^\s\w]', re.UNICODE)
    return re.findall(token, text)

"""

def toWordsCaseSen(text): # splits on white space
  tokens = re.sub("\t|\n|\r", "", text)
  return tokens.split()
"""
# Return a transition matrix built from the gold standard
# Pass in tags for both languages
def getTransitions(tags, lang1, lang2):
  transitions = {lang1: {}, lang2: {}}
  counts = Counter(zip(tags, tags[1:]))

  total = sum(counts.values()) # Get new total for language tags

  for (x, y), c in counts.iteritems(): # Compute transition matrix
    transitions[x][y] = math.log(c / float(total))
  return transitions

class Evaluator:
    def __init__(self, cslm, transitions, tags):
        self.cslm = cslm
        self.transitions = transitions
        self.tags = tags
        self.engClassifier = StanfordNERTagger(
            "../stanford-ner-2015-04-20/classifiers/english.all.3class.distsim.crf.ser.gz",
            "../stanford-ner-2015-04-20/stanford-ner.jar")
        self.spanClassifier = StanfordNERTagger(
            "../stanford-ner-2015-04-20/classifiers/spanish.ancora.distsim.s512.crf.ser.gz",
            "../stanford-ner-2015-04-20/stanford-ner.jar")

    def tagger(self, text_list):
        hmm = HiddenMarkovModel(text_list, self.tags, self.transitions, self.cslm)
        hmmtags = hmm.generateTags() # generate list of hmm tags
        words = hmm.words # generate list of words
        taggedTokens = []
        prevLang = "Eng"
        engTags = []
        spnTags = []
        engTag = ""
        spanTag = ""
        print "Tagging {} words".format(len(words))
        for k, word in enumerate(words):
            # check if punctuation else use hmmtag
            lang = 'Punct' if word in string.punctuation else hmmtags[k]
            lang = 'Num' if word.isdigit() else lang
            # check if word is NE
            if lang != "Punct":
              index = k % 1000
              if index == 0:
                engTags = self.engClassifier.tag(words[k:k+1000])
                spnTags = self.spanClassifier.tag(words[k:k+1000])
              engTag = engTags[index][1]
              spanTag = spnTags[index][1]
            else:
              engTag = "O"
              spanTag = "O"

            # mark as NE if either classifier identifies it
            if engTag != 'O' or spanTag != 'O':
                NE = "{}/{}".format(engTag, spanTag)
            else:
                NE = "O"
            # record probabilities
            if lang in ("Eng", "Spn"):
              hmmProb = round(hmm.transitions[prevLang][lang], 2)
              engProb = round(self.cslm.prob("Eng", word), 2)
              spnProb = round(self.cslm.prob("Spn", word), 2)
              totalProb = (hmmProb + engProb) if lang == "Eng" else (hmmProb + spnProb)
              prevLang = lang
            else:
              hmmProb = "N/A"
              engProb = "N/A"
              spnProb = "N/A"
              totalProb = "N/A"

            taggedTokens.append((word, lang, NE, str(engProb), str(spnProb), str(hmmProb), str(totalProb)))
            #taggedTokens.append((word, lang, NE))
            #print word, lang, NE
        return taggedTokens

    #  Tag testCorpus and write to output file
    def annotate(self, testCorpus):
        print "Annotation Mode"
        with io.open(testCorpus.strip(".txt") + '_annotated.txt', 'w', encoding='utf8') as output:
            text = io.open(testCorpus).read()
            testWords = toWordsCaseSen(text)
            tagged_rows = self.tagger(testWords)
            output.write(u"Token\tLanguage\tNamed Entity\tEng-NGram Prob\tSpn-NGram Prob\tHMM Prob\tTotal Prob\n")
            for row in tagged_rows:
                csv_row = '\t'.join([unicode(s) for s in row]) + u"\n"
                print csv_row
                output.write(csv_row)
            print "Annotation file written"

    #  Evaluate goldStandard and write to output file
    def evaluate(self, goldStandard):
        print "Evaluation Mode"
        with io.open(goldStandard + '_outputwithHMM.txt', 'w', encoding='utf8') as output:

            #create list of text and tags
            lines = io.open(goldStandard, 'r', encoding='utf8').readlines()
            text, gold_tags = [], []
            for x in lines:
                columns = x.split("\t")
                text.append(columns[-2].strip())
                gold_tags.append(columns[-1].strip())


            # annotate text with model
            annotated_output = self.tagger(text)
            #tokens, lang_tags, NE_tags = map(list, zip(*annotated_output))
            tokens, lang_tags, NE_tags, engProbs, spnProbs, hmmProbs, totalProbs = map(list, zip(*annotated_output))

            # set counters to 0
            langCorrect = langTotal = NECorrect = NETotal = 0
            evaluations = []

            # compare gold standard and model tags
            for lang, NE, gold in zip(lang_tags, NE_tags, gold_tags):
                if gold in ('Eng', 'Spn'):   #evaluate language tags
                    langTotal += 1
                    if gold == lang:
                        langCorrect += 1
                        evaluations.append("Correct")
                    else:
                        evaluations.append("Incorrect")
                # evaluate NE tags
                elif gold == "NamedEnt":
                    NETotal += 1
                    if NE != 'O':
                        NECorrect += 1
                        evaluations.append("Correct")
                    else:
                        evaluations.append("Incorrect")
                # don't evaluate punctuation
                else:
                    evaluations.append("NA")
            #write
            output.write(u"Language Accuracy: {}\n".format(langCorrect / float(langTotal)))
            output.write(u"NE Accuracy: {}\n".format(NECorrect / float(NETotal)))
            output.write(u"Token\tGold Standard\tTagged Language\tNamed Entity\tEvaluation\n")
            for all_columns in zip(text, gold_tags, lang_tags, NE_tags, evaluations):
                output.write(u"\t".join(all_columns) + u"\n")
            print "Evaluation file written"

"""
Process arguments
Get corpora and create NGram models
Create Code-Switch Language Model
Build Markov model with Expectation Minimization
Annotate
Evaluate
"""
# Evaluation.py goldStandard testCorpus
def main(argv):
    goldStandard = io.open(argv[0], 'r', encoding='utf8')
    #testCorpus = io.open(argv[1], 'r', encoding='utf8')
    n = 5
    #engData = toWords(io.open('./TrainingCorpora/Subtlex.US.trim.txt', 'r', encoding='utf8').read())
    engData = toWords(io.open("./TrainingCorpora/EngCorpus-1m.txt",'r', encoding='utf8').read())
    #spnData = toWords(io.open('./TrainingCorpora/ActivEsCorpus.txt', 'r', encoding='utf8').read())
    spnData = toWords(io.open('./TrainingCorpora/MexCorpus.txt', 'r', encoding='utf8').read())
    enModel = CharNGram('Eng', getConditionalCounts(engData, n), n)
    esModel = CharNGram('Spn', getConditionalCounts(spnData, n), n)

    cslm = CodeSwitchedLanguageModel([enModel, esModel])

    tags = [u"Eng", u"Spn"]

    # Split on tabs and extract the gold standard tag
    goldTags = [x.split("\t")[-1].strip() for x in goldStandard.readlines()]
    otherSpn = ["NonStSpn", "SpnNoSpace"]
    otherEng = ["NonStEng", "EngNoSpace", "EngNonSt"]

    # Convert all tags to either Eng or Spn and remove others
    goldTags = ["Eng" if x in otherEng else x for x in goldTags]
    goldTags = ["Spn" if x in otherSpn else x for x in goldTags]
    goldTags = [x for x in goldTags if x in ("Eng", "Spn")]

    # Compute prior based on gold standard
    transitions = getTransitions(goldTags, tags[0], tags[1])

    eval = Evaluator(cslm, transitions, tags)
    eval.annotate(argv[1])
    eval.evaluate(argv[0])

    #  Use an array of arguments?
    #  Should user pass in number of characters, number of languages, names of
    #  languages?

if __name__ == "__main__":
    main(sys.argv[1:]) # Skip over script name
