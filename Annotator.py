# Annotator.py
# Using Python 2.7.11

import CharNGram
import io

class Annotator:
  def __init__(self, hmm):
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
