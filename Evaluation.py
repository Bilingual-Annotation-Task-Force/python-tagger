# Evaluation.py
# Using Python 2.7.11

import sys
import getopt
import re
import CharNGram
import io

""" Splits text input into words and formats them, splitting by whitespace

    @param lines a list of text lines 
    @return a list of lists of formatted words
"""

def toWords(lines):
  for i, line in enumerate(lines):
    line = re.sub('[\W+]', "", line)
    lines[i] = line.lower().split(" ")

class Evaluator:
  def __init__(self, cslm, hmm):
    self.cslm = cslm
    self.hmm = hmm
    self.engClassifier = 0
    self.spanClassifier = 0
    # Dict of classifiers with name as key and value as classifier?

  # Write output to file
  def annotate(self, filename):
    with io.open(filename + '_annotated.txt', 'w', encoding = 'utf8') as output:
      output.write('Token, Tag')
      hmmtags = self.hmm.generateTags()
      words = self.hmm.words

      for k in xrange(len(words)):
        guess = hmmtags[k]
        word = words[k]

        if word.match('\\p{P}'):
          guess = 'Punct'

        engClassification = self.engClassifier.classiftyWithInlineXML(word)
        spanClassification = self.spanClassifier.classiftyWithInlineXML(word)

        if '<' in engClassification and guess == 'Eng':
          guess = 'EngNamedEnt'

        if '<' in spanClassification and guess == 'Spn':
          guess = 'SpnNamedEnt'

        output.write(word + ',' + guess)

      output.close()

  # Write evaluation of annotation to file
  def evaluate(self, goldStandard):
    with io.open(goldStandard + '_outputwithHMM.txt', 'w', encoding = 'utf8') as output:
      output.write('Word \t Guess \t Tag \t Correct/Incorrect')
      lines = io.open(goldStandard, 'r', encoding = 'utf8').readlines()
      hmmtags = self.hmm.generateTags()

      correct = 0
      total = 0

      for k in xrange(len(lines)):
        annotation = lines[k].split('\t')
        word = annotation[0]
        tag = annotation[1]
        guess = hmmtags[k]

        if word.match('\\p{P}'):
          guess = 'Punct'

        engClassification = self.engClassifier.classiftyWithInlineXML(word)
        if '<' in engClassification:
          guess = 'Named Ent'

        outputFile.write(word + ',' + guess + ',' + tag)

        if tag == 'Eng' or tag == 'Spn' or tag == 'Named Ent':
          if tag == 'Eng' and guess == 'Eng':
            correct += 1

          elif tag == 'Spn' and guess == 'Spn':
            correct += 1

          elif tag == 'Named Ent' and guess == 'Named Ent':
            correct += 1

          else:
            output.write('\t INCORRECT')

          total += 1

        output.write('\n')

      output.close()
      return correct / total


"""
Process arguments
Get corpora and create NGram models
Create Code-Switch Language Model
Build Markov model with Expectation Minimization
Annotate
Evaluate
"""
def main(argv = sys.argv):
  testCorpus = 'INSERT RELATIVE PATH HERE' # Extract from arguments?
  goldStandard = 'INSERT RELATIVE PATH HERE'

  n = 5
  engData = toWords(io.open('PATH TO ENG DATA', 'r', encoding = 'utf8').readlines())
  spanData = toWords(io.open('PATH TO SPAN DATA', 'r', encoding = 'utf8').readlines())
  enModel = NGramModel('Eng', getConditionalCounts(engData, n), n)
  esModel = NGramModel('Spn', getConditionalCounts(spanData, n), n)

  cslm = CodeSwitchedLanguageModel([enModel, esModel])

  testWords = 0

  tags = ['Eng', 'Spn']

  transitions = 0 # Insert Expectation Maximization call here
  hmm = HiddenMarkovModel(testWords, tags, transitions, cslm)

  eval = Evaluator(cslm, hmm)
  eval.annotate(testCorpus)

  # Use an array of arguments?
  # Should user pass in number of characters, number of languages, names of
  # languages?

if __name__ == "__main__":
  main()

