# Evaluation.py

import re

""" Splits text input into words and formats them, splitting by whitespace

    @param lines a list of text lines 
    @return a list of lists of formatted words
"""

def toWords(lines):
  for i, line in enumerate(lines):
    line = re.sub('[\W+]', "", line)
    lines[i] = line.lower().split(" ")

