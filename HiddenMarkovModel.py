# HiddenMarkovModel.py
# Using Python 2.7.11

import math

class HiddenMarkovModel:
  def __init__(self, words, tagSet, transitions, cslm):
    self.words = words
    self.tagSet = tagSet
    self.transitions = transitions
    self.cslm = cslm
    self.v = [[0 for i in xrange(len(words))] for j in xrange(len(tagSet))]

  # Run Viterbi algorithm and retrace to compute most likely transitions
  def generateTags(self):
    self.viterbi()
    self.retrace()

  # Return emission probability
  def em(self, ctx, word):
    return self.cslm.prob(ctx, word)

  # Return transmission probability
  def tr(self, ctx, tag):
    return self.transitions[ctx][tag]

  def viterbi(self):
    # Equal probability of starting with either tag
    for k, tag in enumerate(self.tagSet):
      self.v[0][k] = Node(math.log(0.5), k)

    for wordIndex, word in enumerate(self.words, start=1):
      for tagIndex, tag in enumerate(self.tagSet):
        # Using map
        # transitionProbs = map(lambda x: Node(self.v[wordIndex - 1][x].prob +
        #       math.log(self.tr(self.tagSet[x], self.tagSet[tagIndex])), x),
        #       xrange(len(self.tagSet)))

        # Using list comprehension
        # transitionProbs = [Node(self.v[wordIndex - 1][x].prob +
        #   math.log(tr(self.tagSet[x], self.tagSet[tagIndex])), x) for x in
        #   xrange(len(self.tagSet))]

        # Using loop
        transitionProbs = []
        for x, tag in enumerate(self.tagSet):
          transitionProbs.append(Node(self.v[wordIndex - 1][x].prob +
            math.log(self.tr(self.tagSet[x], self.tagSet[tagIndex])), x))


        maxNode = reduce(lambda n1, n2: n1 if n1.prob > n2.prob else n2, transitionProbs)
        emissionProb = em(self.tagSet[tagIndex], self.words[wordIndex])
        self.v[wordIndex][tagIndex] = Node(math.log(emissionProb) +
            maxNode.prob, maxNode.prevTag)
        pass

  def retrace(self):
    tags = [0 for i in xrange(len(self.words))]

    # Find most probable final tag

    # Using reduce
    # last = reduce(lambda x, y: x if self.v[len(self.words) - 1][x].prob >
    #        self.v[len(self.words) - 1][y] else y, xrange(len(self.tagSet)))

    # Using loops
    last = 0
    for x, taglist in enumerate(self.tagSet):
      for y, tag in enumerate(taglist):
        if self.v[len(self.words) - 1][x].prob > self.v[len(self.words) - 1][y]:
          last = x
        else:
          last = y

    tags[len(self.words) - 1] = self.tagSet[last]

    # Follow backpointers to most probable previous tags
    prev = self.v[len(self.words) - 1][last].prev
    for k in xrange(len(self.words) - 2, 0, -1):
      tags[k] = self.tagSet[prev]
      prev = self.v[k][prev].prev

    return tags

class Node:
  def __init__(self, prob, prevTag):
    self.prob = prob
    self.prevTag = prevTag

