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

""" Splits text input into words and formats them, splitting by whitespace

    @param text a string of text
    @return a list of formatted words
"""
# case-insensitive tokenizer for ngram probabilities only

"""
def toWords(text):
    # requires utf-8 encoding
    token = re.compile(ur'[\w]+|[^\s\w]', re.UNICODE)
    tokens = re.findall(token, text)
    return [word.lower() for word in tokens]
    """

def toWords(text):
    tokens = re.sub("\t|\n|\r", "", text)
    return [word.lower() for word in tokens.split()]

"""
def toWordsCaseSen(text):
    # requires utf-8 encoding
    token = re.compile(ur'[\w]+|[^\s\w]', re.UNICODE)
    return re.findall(token, text)
    """

def toWordsCaseSen(text):
  tokens = re.sub("\t|\n|\r", "", text)
  return tokens.split()

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
    def __init__(self, cslm, hmm):
        self.cslm = cslm
        self.hmm = hmm
        self.engClassifier = StanfordNERTagger(
            "../stanford-ner-2015-04-20/classifiers/english.all.3class.distsim.crf.ser.gz",
            "../stanford-ner-2015-04-20/stanford-ner.jar")
        self.spanClassifier = StanfordNERTagger(
            "../stanford-ner-2015-04-20/classifiers/spanish.ancora.distsim.s512.crf.ser.gz",
            "../stanford-ner-2015-04-20/stanford-ner.jar")

    def tagger(self):
        hmmtags = self.hmm.generateTags()
        words = self.hmm.words  # this needs to be case-sensitive
        taggedTokens = [("Token", "Language", "Named Entity", "Eng-NGram Prob",
          "Spn-NGram Prob", "HMM Prob", "Total Prob")]

        prevLang = "Eng"
        for k, word in enumerate(words):

            # check if punctuation else use hmmtag
            lang = 'Punct' if word in string.punctuation else hmmtags[k]
            lang = 'Num' if word.isdigit() else lang

            # check if word is NE
            """
            try:
                engTag = self.engClassifier.tag(words[k-2:k+2])[2][1]
                spanTag = self.spanClassifier.tag(words[k-2:k+2])[2][1]

            except IndexError:
                engTag = self.engClassifier.tag([word])[0][1]
                spanTag = self.spanClassifier.tag([word])[0][1]
              """

            if lang != "Punct":
              if lang == "Eng":
                engTag = self.engClassifier.tag([word])[0][1]
                spanTag = "O"
              else:
                spanTag = self.spanClassifier.tag([word])[0][1]
                engTag = "O"
            else:
              engTag = "O"
              spanTag = "O"

            # mark as NE if either classifier identifies it
            if engTag != 'O' or spanTag != 'O':
                NE = "{}/{}".format(engTag, spanTag)
            else:
                NE = "O"

            if lang in ("Eng", "Spn"):
              hmmProb = self.hmm.transitions[prevLang][lang]
              engProb = self.hmm.cslm.prob("Eng", word)
              spnProb = self.hmm.cslm.prob("Spn", word)
              totalProb = (hmmProb + engProb) if lang == "Eng" else (hmmProb + spnProb)
              prevLang = lang
            else:
              hmmProb = "N/A"
              engProb = "N/A"
              spnProb = "N/A"
              totalProb = "N/A"


            taggedTokens.append((word, lang, NE, engProb, spnProb, hmmProb, totalProb))
            print k, word, lang, NE, engProb, spnProb, hmmProb, totalProb
        return taggedTokens

    #  Write annotation to output file
    def annotate(self, textfile):
        with io.open(textfile + '_annotated.txt', 'w', encoding='utf8') as output:
            text = io.open(textfile).read()

            hmmtags = self.hmm.generateTags()
            words = self.hmm.words  # this needs to be case-sensitive
            # taggedTokens = [("Token", "Language", "Named Entity", "Eng-NGram Prob",
            #  "Spn-NGram Prob", "HMM Prob")]
            output.write(u"Token\tLanguage\tNamed Entity\tEng-NGram Prob\tSpn-NGram Prob\tHMM Prob\tTotal Prob\n")
            print "Token\tLanguage\tNamed Entity\tEng-NGram Prob\tSpn-NGram Prob\tHMM Prob\tTotal Prob"
            prevLang = "Eng"

            engTags = []
            spnTags = []
            engTag = ""
            spanTag = ""

            for k, word in enumerate(words):

                # check if punctuation else use hmmtag
                lang = 'Punct' if word in string.punctuation else hmmtags[k]
                lang = 'Num' if word.isdigit() else lang

                # check if word is NE
                """
                try:
                    engTag = self.engClassifier.tag(words[k-2:k+2])[2][1]
                    spanTag = self.spanClassifier.tag(words[k-2:k+2])[2][1]

                except IndexError:
                    engTag = self.engClassifier.tag([word])[0][1]
                    spanTag = self.spanClassifier.tag([word])[0][1]
                """

                # Get context from next five words
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
                # First, try always English, then always Spanish, then the language
                if engTag != 'O' or spanTag != 'O':
                    NE = "{}/{}".format(engTag, spanTag)
                    # if lang == "Eng":
                    #     NE = "{}".format(spanTag)
                    # else:
                    #     NE = "{}".format(engTag)

                else:
                    NE = "O"

                if lang in ("Eng", "Spn"):
                  hmmProb = self.hmm.transitions[prevLang][lang]
                  engProb = self.hmm.cslm.prob("Eng", word)
                  spnProb = self.hmm.cslm.prob("Spn", word)
                  totalProb = (hmmProb + engProb) if lang == "Eng" else (hmmProb + spnProb)
                  prevLang = lang
                else:
                  hmmProb = "N/A"
                  engProb = "N/A"
                  spnProb = "N/A"
                  totalProb = "N/A"

                # taggedTokens.append((word, lang, NE, engProb, spnProb, hmmProb))
                output.write(u"{}\t{}\t{}\t{}\t{}\t{}\t{}\n".format(word, lang, NE, engProb, spnProb, hmmProb, totalProb))
                print k, word, lang, NE, engProb, spnProb, hmmProb, totalProb

    #  Write evaluation of annotation to file
    def evaluate(self, goldStandard, textfile):
        with io.open(goldStandard + '_outputwithHMM.txt', 'w', encoding='utf8') as output:
            lines = io.open(goldStandard, 'r', encoding='utf8').readlines()
            text = [x.split("\t")[-2].strip() for x in lines]
            gold_tags = [x.split("\t")[-1].strip() for x in lines]
            #annotated_output = io.open(textfile + "_annotated.txt", "r", encoding="utf8").readlines()[1:]
            annotated_output = tagger(text)
            lang_tags = [x.split("\t")[1].strip() for x in annotated_output]
            ne_tags = [x.split("\t")[2].strip() for x in annotated_output]
            langCorrect = langTotal = NECorrect = NETotal = 0
            evaluations = []

            # compare gold standard and model tags
            for word, gold, lang, NE in zip(text, gold_tags, lang_tags, ne_tags):
                print word, gold, lang, NE

                # evaluate language tags
                if gold in ('Eng', 'Spn'):
                    langTotal += 1
                    if gold == 'Eng' and lang == 'Eng':
                        langCorrect += 1
                        evaluations.append("Correct")
                    elif gold == 'Spn' and lang == 'Spn':
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

            output.write(u"Language Accuracy: {}\n".format(langCorrect / float(langTotal)))
            output.write(u"NE Accuracy: {}\n".format(NECorrect / float(NETotal)))
            output.write(u"Token\tGold Standard\tTagged Language\tNamed Entity\tEvaluation\n")
            for word, gold, lang, NE, evals in zip(text, gold_tags, lang_tags, ne_tags, evaluations):
                output.write(u"{}\t{}\t{}\t{}\t{}\n".format(word, gold, lang, NE, evals))

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
    testCorpus = io.open(argv[1], 'r', encoding='utf8')
    n = 5
    #engData = toWords(io.open('./TrainingCorpora/Subtlex.US.trim.txt', 'r', encoding='utf8').read())
    engData = toWords(io.open("./TrainingCorpora/EngCorpus-1m.txt",'r', encoding='utf8').read())
    #spnData = toWords(io.open('./TrainingCorpora/ActivEsCorpus.txt', 'r', encoding='utf8').read())
    spnData = toWords(io.open('./TrainingCorpora/MexCorpus.txt', 'r', encoding='utf8').read())
    enModel = CharNGram('Eng', getConditionalCounts(engData, n), n)
    esModel = CharNGram('Spn', getConditionalCounts(spnData, n), n)

    cslm = CodeSwitchedLanguageModel([enModel, esModel])

    testWords = toWordsCaseSen(testCorpus.read())
    # testWords = [word.strip() for word in testCorpus.readlines()]

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
    hmm = HiddenMarkovModel(testWords, tags, transitions, cslm)

    eval = Evaluator(cslm, hmm)
    eval.annotate(argv[1])
    eval.evaluate(argv[0], argv[1])

    #  Use an array of arguments?
    #  Should user pass in number of characters, number of languages, names of
    #  languages?

if __name__ == "__main__":
    main(sys.argv[1:]) # Skip over script name
