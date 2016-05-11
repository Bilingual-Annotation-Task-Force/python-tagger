# Evaluation.py
# Using Python 2.7.11

import sys
import getopt
import re
import io
import HiddenMarkovModel
import Annotator

""" Splits text input into words and formats them, splitting by whitespace

    @param lines a list of text lines 
    @return a list of lists of formatted words
"""

def toWords(lines):
  for i, line in enumerate(lines):
    tokens = re.compile(r'[\w]+|[^\s\w]', re.UNICODE)
    line = re.findall(tokens, line) #create a list of tokens from the line
    lines[i] = [word.lower() for word in line]

class Evaluator:
  def __init__(self, cslm, hmm):
    self.cslm = cslm
    self.hmm = hmm
    self.engClassifier = 0
    self.spanClassifier = 0

  # Write output to file
  def annotate(self, filename):
    with io.open(filename + '_annotated.txt', 'w', encoding='utf-8') as output:
      output.write('Token, Tag\n')
      hmmtags = self.hmm.generateTags()
      words = self.hmm.words

      for k, word in enumerate(words):
        guess = hmmtags[k]

        if word.match('\\p{P}'):
          guess = 'Punct'
        
        word_context = " ".join(words[k-3:k+3])
        engTag = self.engClassifier.tag(word_context)[3]
        spanTag = self.spanClassifier.tag(word_context)[3]

        if engTag[1] != 'O' and guess == 'Eng':
            guess = engTag[1]

        if spanTag[1] != 'O' and guess == 'Spn':
          guess = spanTag[1]

        output.write(word + ',' + guess + '\n')

  # Write evaluation of annotation to file
  def evaluate(self, goldStandard):
    with io.open(goldStandard + '_outputwithHMM.txt', 'w', encoding='utf8') as output:
      output.write('Word \t Guess \t Tag \t Correct/Incorrect\n')
      lines = io.open(goldStandard, 'r', encoding='utf8').readlines()
      hmmtags = self.hmm.generateTags()

      correct = 0
      total = 0

      for k, line in enumerate(lines):
        annotation = line.split('\t')
        word = annotation[0]
        tag = annotation[1]
        guess = hmmtags[k]

        if word.match('\\p{P}'):
          guess = 'Punct'

        engTag = self.engClassifier.tag(word)
        spanTag = self.spanClassifier.tag(word)

        if engTag[1] != 'O' and guess == 'Eng':
          guess = 'EngNamedEnt'

        if spanTag[1] != 'O' and guess == 'Spn':
          guess = 'SpnNamedEnt'

        # CSV or TSV output?
        outputFile.write(word + ',' + guess + ',' + tag)

        # Handling Named Entity case?
        if tag in ('Eng', 'Spn', 'Named Ent'):
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

      return correct / float(total)

"""
Process arguments
Get corpora and create NGram models
Create Code-Switch Language Model
Build Markov model with Expectation Minimization
Annotate
Evaluate
"""
def main(argv=sys.argv):
  testCorpus = 'INSERT RELATIVE PATH HERE' # Extract from arguments?
  goldStandard = 'INSERT RELATIVE PATH HERE'

  n = 5
  engData = toWords(io.open('PATH TO ENG DATA', 'r', encoding='utf8').readlines())
  spanData = toWords(io.open('PATH TO SPAN DATA', 'r', encoding='utf8').readlines())
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

