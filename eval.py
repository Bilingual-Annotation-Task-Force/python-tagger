#  Evaluation.py
#  Using Python 2.7.11
#trial line August 11, 2016
import sys
import io
import string
import re

from collections import Counter

import csv
import math

from nltk.tag.stanford import StanfordNERTagger

from hmm import HiddenMarkovModel
from cngram import *
from cs_model import CodeSModel


def split_words(text, keep_case = True):
    """Splits a string of white-space separated words into tokens of words

    Args:
        text (str): String containing all words
        keep_case (bool, optional): Determine if the case of the letters
            is maintained. Defaults to True.

    Returns:
        list<str>: List of all the tokens within the .
    """
    if not keep_case:
        text = text.lower()
    token = re.compile(ur'[\w]+|[^\s\w]', re.UNICODE)
    return re.findall(token, text)


def get_transi_matrix(tags, lang1, lang2):
    """Return a transition matrix from the gold standard.

    The transition matrix is a chart matching the probability of every
        transition between two different tags. It counts the number of times a
        language tag stays the same or changes, looking similar to this diagram,
        in row-major order:

                +-----------------+
                |  tag1  |  tag1  |
                |   ->   |   ->   |
                |  tag1  |  tag2  |
                +--------+--------+
                |  tag2  |  tag2  |
                |   ->   |   ->   |
                |  tag1  |  tag2  |
                +-----------------+

    However, it takes the log of the raw, overall probability of each switch to
        magnify small differences.

    Args:
        tags (list<str>): A list of language tags, should only contain two main
            languages.
        lang1 (str): Tag of the first language.
        lang2 (str): Tag of the second language.

    Returns:
        dict<str -> dict<str -> list>>: A dictionary representing the transition
            matrix.
    """
    transi_matrix = {lang1: {}, lang2: {}}
    # Count number of occurences for each distinct transition
    counts = Counter(zip(tags, tags[1:])) # Generating pairs
    total = sum(counts.values()) # Counting
    # Compute the matrix
    for (x, y), c in counts.iteritems():
        # Taking the log to magnify the closer differences
        transi_matrix[x][y] = math.log(c / float(total))
    return transi_matrix


class Evaluator:
    """Evaluates the input files to determine the language of each word.

    This utilizes the effectiveness of two separate models, composing them to
        form a more complete model. The two models being used are the Code
        Switched Language Model and the Hidden Markov Model.

    Args:
        cs_model (CodeSModel): The Code Switched Language Model.
        transi_matrix (list<list<float>>): A matrix of percentages from one tag
            to another. See the function get_transi_matrix for more info about
            the specific format. Should be a 2x2 matrix of values for now.
        tags (list<str>): List of tags.

    Properties:
        cs_model (CodeSModel): A code switched language model. Check the file
            switched_model.py.
        transi_matrix (list<list<float>>): A matrix of percentages from one tag
            to another. See the function get_transi_matrix for more info about
            the specific format.
        tags (list<str>): List of tags matched to the langauge
        lang1_tagger (StanfordNERTagger): The NLTK Tagger for language 1
        lang2_tagger (StanfordNERTagger): The NLTK Tagger for language 2
    """

    def __init__(self, cs_model, transi_matrix, tags):
        self.cs_model = cs_model
        self.transI_matrix = transi_matrix
        self.tags = tags
        self.lang1_tagger = StanfordNERTagger(
            "../stanford-ner-2015-04-20/classifiers/english.all.3class.distsim.crf.ser.gz",
            "../stanford-ner-2015-04-20/stanford-ner.jar")
        self.lang2_tagger = StanfordNERTagger(
            "../stanford-ner-2015-04-20/classifiers/spanish.ancora.distsim.s512.crf.ser.gz",
            "../stanford-ner-2015-04-20/stanford-ner.jar")

    def tag_list(self, word_list):
        """Tagger generates a list of tags, which contains multiple pieces of
            information from many different models and combines them into one
            list.

        The list provided is a tuple of the following elements:
            1. The word itself
            2. The language tagged
            3. Named entity
            4. Probability of lang1_prob
            5. Probability of lang2_prob
            6. Probability of hmm_prob
            7. Probability of total_prob

        The list's order matches the order of word_list.

        Args:
            word_list (list<str>): The list of tokens being processed.

        Return:
            list<tuple<str, str, str, str, str, str, str, str>>:
                Refer to the above for list of entries in the tuple,
                as well as details regarding the list itself.
        """
        # Why is this generated here?
        hmm = HiddenMarkovModel(word_list, self.tags, self.transi_matrix,
            self.cs_model)
        hmmtags = hmm.gen_tags()
        # Is this necessary? Does the hmm do anything to the text?
        words = hmm.words

        tagged_tokens = []
        prev_lang = "Eng"
        lang1_tags = []
        lang2_tags = []
        lang1_tag = "Eng"
        lang2_tag = "Spn"
        token = re.compile(ur'[^\w\s]', re.UNICODE)
        # Tag each word based on the statistical analysis
        print("Tagging {} words".format(len(words)))
        for k, word in enumerate(words):
            # check if punctuation or number, otherwise use tag from hmm
            if re.match(token, word) and not word[-1].isalpha():
                lang = 'Punct'
            elif word.isdigit():
                lang = 'Num'
            else:
                lang = hmmtags[k]
            # check if word is a named entity
            if lang != "Punct":
                # Processing 1000 words at a time
                index = k % 1000
                if index == 0:
                    lang1_tags = self.lang1_tagger.tag(words[k:k+1000])
                    lang2_tags = self.lang2_tagger.tag(words[k:k+1000])
                lang1_tag = lang1_tags[index][1]
                lang2_tag = lang2_tags[index][1]
            else:
              lang1_tag = "O"
              lang2_tag = "O"
            # mark as NE if either NLTK tagger identifies it
            if lang1_tag != 'O' or lang2_tag != 'O':
                NE = "{}/{}".format(lang1_tag, lang2_tag)
            else:
                NE = "O"
            # record probabilities
            if lang in (lang1_tag, lang2_tag):
                hmm_prob = round(hmm.transitions[prev_lang][lang], 2)
                lang1_prob = round(self.cs_model.prob(lang1_tag, word), 2)
                lang2_prob = round(self.cs_model.prob(lang2_tag, word), 2)
                if lang == lang1_tag:
                    total_prob = (hmm_prob + lang1_prob)
                else:
                    total_prob = (hmm_prob + lang2_prob)
                prev_lang = lang
            else:
                hmm_prob = "N/A"
                lang1_prob = "N/A"
                lang2_prob = "N/A"
                total_prob = "N/A"

            taggedTokens.append((word, lang, NE,
                str(lang1_prob), str(lang2_prob), str(hmm_prob),
                str(total_prob)))

        return taggedTokens

    def annotate(self, corpus):
        """Annotates a corpus by adding tags for the words of the corpus.
        Then prints the generated tags and data to a new .csv file.
        The new .csv files shoall be named as such:

            <corpus_name>_annotated.txt

        where <corpus_name> is replaced with the name of the corpus file
            without the file extension.

        Args:
            corpus (str): The path to the corpus file
        """
        print("Annotation Mode")
        with io.open(corpus.strip(".txt") + '_annotated.txt',
                'w', encoding='utf8') as output:
            text = io.open(corpus).read()
            testWords = split_words(text)
            tagged_rows = self.tag_list(testWords)
            output.write(u"Token\tLanguage\tNamed Entity"
                u"\tEng-NGram Prob\tSpn-NGram Prob"
                u"\tHMM Prob\tTotal Prob\n")
            for row in tagged_rows:
                csv_row = '\t'.join([unicode(s) for s in row]) + u"\n"
                print(csv_row)
                output.write(csv_row)
            print("Annotation file written")

    def evaluate(self, gold_standard):
        """Evaluates the system, comparing system output to the gold standard's
            tags. The final file will be:

        <gold_standard>_outputwithHMM.txt

        Args:
            gold_standard (str): The path to the gold standard
        """
        # Output to file
        print("Evaluation Mode")
        with io.open(gold_standard + '_outputwithHMM.txt',
                'w', encoding='utf8') as output:
            #create list of text and tags
            lines = io.open(gold_standard, 'r', encoding='utf8').readlines()
            text, gold_tags = [], []
            # Store words/token into text, and tags into gold_tags
            for x in lines:
                columns = x.split("\t")
                text.append(columns[-2].strip())
                gold_tags.append(columns[-1].strip())
            # Tag the text based on the provided models
            annotated_output = self.tag_list(text)
            tokens, lang_tags, NE_tags, lang1_probs, lang2_probs,\
                    hmm_probs, total_probs = map(list, zip(*annotated_output))
            # Reset counters to 0, prepare for checking with the gold_standard
            langCorrect = langTotal = NECorrect = NETotal = 0
            evals = []
            # Compare gold standard and model tags
            for lang, NE, gold in zip(lang_tags, NE_tags, gold_tags):
                # Evaluate language tags
                if gold in ('Eng', 'Spn'):
                    langTotal += 1
                    if gold == lang:
                        langCorrect += 1
                        evals.append("Correct")
                    else:
                        evals.append("Incorrect")
                # Evaluate NE tags
                elif gold == "NamedEnt":
                    NETotal += 1
                    if NE != 'O':
                        NECorrect += 1
                        evals.append("Correct")
                    else:
                        evals.append("Incorrect")
                # Don't evaluate punctuation
                else:
                    evals.append("NA")
            # Write the final results to file
            output.write(u"Language Accuracy: {}\n".format(
                langCorrect / float(langTotal)))
            output.write(u"NE Accuracy: {}\n".format(
                NECorrect / float(NETotal)))
            output.write(u"Token\tGold Standard\tTagged Language"
                "\tNamed Entity\tEvaluation\n")
            for all_columns in zip(text, gold_tags, lang_tags, NE_tags, evals):
                output.write(u"\t".join(all_columns) + u"\n")
            print("Evaluation file written")


# This method needs to be rewritten
# eval.py gold_standard test_corpus
def main(argv):
    """Main prep work and evaluation. Process:
    1. Process arguments
    2. Get corpora
    3. Create NGram models
    4. Create Code-Switch Model
    5. Build Markov model with Expectation Minimization
    6. Annotate
    7. Evaluate

    NOTE: to be swapped with a more dynamic system in the future
    """
    gold_standard = io.open(argv[0], 'r', encoding='utf8')
    # testCorpus = io.open(argv[1], 'r', encoding='utf8')
    n = 5
    # lang1_data = toWords(io.open('./TrainingCorpora/Subtlex.US.trim.txt',
    #   'r', encoding='utf8').read())
    lang1_data = split_words(io.open("./TrainingCorpora/EngCorpus-1m.txt",
        'r', encoding='utf8').read())
    # lang2_data = toWords(io.open('./TrainingCorpora/ActivEsCorpus.txt',
    #   'r', encoding='utf8').read())
    lang2_data = split_words(io.open('./TrainingCorpora/MexCorpus.txt',
        'r', encoding='utf8').read())
    lang1_model = CNGram('Eng', get_cond_cnts(lang1_data, n), n)
    lang2_model = CNGram('Spn', get_cond_cnts(lang2_data, n), n)

    cs_model = CodeSModel([lang1_model, lang2_model])

    tags = [u"Eng", u"Spn"]

    # Split on tabs and extract the gold standard tag
    gold_tags = [x.split("\t")[-1].strip() for x in gold_standard.readlines()]
    otherlang2 = ["NonStSpn", "SpnNoSpace"]
    otherlang1 = ["NonStEng", "EngNoSpace", "EngNonSt"]

    # Convert all tags to either lang1_ or lang2_ and remove others
    gold_tags = ["Eng" if x in otherlang1 else x for x in gold_tags]
    gold_tags = ["Spn" if x in otherlang2 else x for x in gold_tags]
    gold_tags = [x for x in gold_tags if x in ("Eng", "Spn")]

    # Compute prior based on gold standard
    transitions = get_transi_matrix(gold_tags, tags[0], tags[1])

    eval = Evaluator(cs_model, transitions, tags)
    eval.annotate(argv[1])
    eval.evaluate(argv[0])

    #  Use an array of arguments?
    #  Should user pass in number of characters, number of languages, names of
    #  languages?

if __name__ == "__main__":
    """Sends off the arguments to the method "main" if launched from the
        command line. Sends everything but the first arguments, as the
        first is the name of the script.
    """
    main(sys.argv[1:]) # Skip over script name
