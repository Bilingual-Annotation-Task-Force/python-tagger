# HiddenMarkovModel.py
# Using Python 2.7.11

class HiddenMarkovModel:
  def __init__(self, words, tagSet, transitions, cslm):
    self.words = words
    self.tagSet = tagSet
    self.transitions = transitions
    self.cslm = cslm
    self.v = [[0 for i in xrange(len(words))] for j in xrange(len(tagSet))]

  # Run Viterbi algorithm and retrace to compute most likely transitions
  def generateTags(self):
    viterbi()
    retrace()

  # Return emission probability
  def em(self, ctx, word):
    return cslm.prob(ctx, word)

  # Return transmission probability
  def tr(self, ctx, tag):
    return transitions[ctx][tag]

  def viterbi(self):
    return 0.0

  def retrace(self):
    return 0.0

  # Need data structure to hold prob, prevTag
  # Inner Classes in Python?

