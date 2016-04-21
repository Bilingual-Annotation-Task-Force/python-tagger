# Annotator.py
# Using Python 2.7.11

import io
import HiddenMarkovModel
from nltk.tag.stanford import StanfordNERTagger
class Annotator:
  def __init__(self, hmm):
    self.hmm = hmm
    self.engClassifier = StanfordNERTagger("../stanford-ner-2015-04-20/classifiers/english.all.3class.distsim.crf.ser.gz",
        "../stanford-ner-2015-04-20/stanford-ner.jar")
    self.spanClassifier = StanfordNERTagger("../stanford-ner-2015-04-20/classifiers/spanish.ancora.distsim.s512.crf.ser.gz",
        "../stanford-ner-2015-04-20/stanford-ner.jar")

  # Write output to file
  def annotate(self, filename):
    with io.open(filename + '_annotated.txt', 'w', encoding='utf-8') as output:
      output.write('Token, Tag')
      hmmtags = self.hmm.generateTags()
      words = self.hmm.words

      for k in xrange(len(words)): #to make more pythonic, iterate directly over words: for k in words
        guess = hmmtags[k]
        word = words[k]

        if word.match('\\p{P}'):
          guess = 'Punct'

        engTag = self.engClassifier.tag(words[k])
        spanTag = self.spanClassifier.tag(words[k])

        if engTag[1] != 'O' and guess == 'Eng':
            guess = 'EngNamedEnt'

        if spanTag[1] != 'O' and guess == 'Spn':
          guess = 'SpnNamedEnt'

        output.write(word + ',' + guess)
