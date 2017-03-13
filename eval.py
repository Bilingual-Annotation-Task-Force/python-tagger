#!/usr/bin/env python3
#  eval.py
#  Using Python 3.4.3

import os
import re
import csv
import sys
import math
import argparse
from cngram import CNGram
from cs_model import CodeSModel
from hmm import HiddenMarkovModel
from configparser import ConfigParser
from collections import Counter
from nltk.tag.stanford import StanfordNERTagger


CONFIGS = {}
VERBOSE = False
HEADER = False
TOKENIZE = False


def split_words(text, keep_case=True):
    """Splits a string of white-space separated words into tokens of words

    Args:
        text (str): String containing all words
        keep_case (bool, optional): Determine if the case of the letters
            is maintained. Defaults to True.

    Returns:
        list<str>: List of all the tokens within the text
    """
    if not keep_case:
        text = text.lower()
    token = re.compile(r'[\w]+|[^\s\w]', re.UNICODE)
    return re.findall(token, text)


def get_transi_matrix(gold_tags, langs):
    """Return a transition matrix from the gold standard.

    The transition matrix is a chart matching the probability of every
        transition between two different tags. It counts the number of times a
        language tag stays the same or changes, looking similar to this
        diagram, in row-major order:

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
        gold_tags (list<str>): A list of language tags, should only contain two main
            languages.
        lang1 (str): Tag of the first language.
        lang2 (str): Tag of the second language.

    Returns:
        dict<str -> dict<str -> list>>: A dictionary representing the
        transition matrix.
    """
    transi_matrix = {lang: {} for lang in langs}

    # Count number of occurrences for each distinct transition between tags
    bigram_counts = Counter(zip(gold_tags, gold_tags[1:]))
    total = sum(bigram_counts.values())

    # Compute and normalize the transition matrix
    for (x, y), c in bigram_counts.items():
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
            the specific format.
        tags (list<str>): List of tags.

    Properties:
        cs_model (CodeSModel): A code switched language model. Check the file
            switched_model.py.
        transi_matrix (list<list<float>>): A matrix of percentages from one tag
            to another. See the function get_transi_matrix for more info about
            the specific format.
        tags (list<str>): List of tags matched to the langauge
        lang1_tagger (StanfordNERTagger): The Stanford NER Tagger for language 1
        lang2_tagger (StanfordNERTagger): The Stanford NER Tagger for language 2
    """

    def __init__(self, cs_model, transi_matrix, tags):
        self.cs_model = cs_model
        self.transi_matrix = transi_matrix
        self.tags = tags
        self.lang1_tagger = StanfordNERTagger(CONFIGS["lang1_class"],
                                              CONFIGS["class_jar"])
        self.lang2_tagger = StanfordNERTagger(CONFIGS["lang2_class"],
                                              CONFIGS["class_jar"])

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
        words = hmm.words

        tagged_tokens = []
        lang1_tags = []
        lang2_tags = []
        prev_lang = self.tags[0]
        lang1_tag = self.tags[0]
        lang2_tag = self.tags[1]

        token = re.compile(r'[^\w\s]', re.UNICODE)

        # Tag each word based on ngram model and hmm
        for k, word in enumerate(words):
            # Check if punctuation or number, otherwise use tag from hmm
            if re.match(token, word) and not word[-1].isalpha():
                lang = 'Punct'
            elif word.isdigit():
                lang = 'Num'
            else:
                lang = hmmtags[k]

            # Processing chunks of words at a time
            chunk_size = CONFIGS["ner_chunk_size"]
            index = k % chunk_size

            if index == 0:
                lang1_tags = self.lang1_tagger.tag(words[k:k+chunk_size])
                lang2_tags = self.lang2_tagger.tag(words[k:k+chunk_size])

            lang1_tag = lang1_tags[index][1]
            lang2_tag = lang2_tags[index][1]

            if lang == "Punct":
                lang1_tag = "O"
                lang2_tag = "O"

            # Mark as NE if either NER tagger identifies it
            if lang1_tag != 'O' or lang2_tag != 'O':
                NE = "{}/{}".format(lang1_tag, lang2_tag)
            else:
                NE = "O"

            # Record probabilities
            if lang in CONFIGS["lang_set"]:
                hmm_prob = round(hmm.transi_matrix[prev_lang][lang], 2)
                lang1_prob = round(self.cs_model.prob(self.tags[0], word), 2)
                lang2_prob = round(self.cs_model.prob(self.tags[1], word), 2)
                if lang == self.tags[0]:
                    total_prob = (hmm_prob + lang1_prob)
                else:
                    total_prob = (hmm_prob + lang2_prob)
                prev_lang = lang
            else:
                hmm_prob = "N/A"
                lang1_prob = "N/A"
                lang2_prob = "N/A"
                total_prob = "N/A"

            tagged_tokens.append((word, lang, NE,
                                 str(lang1_prob), str(lang2_prob),
                                 str(hmm_prob), str(total_prob)))

        return tagged_tokens


    def annotate(self, corpus):
        """Annotates a corpus by adding tags for the words of the corpus.
        Then prints the generated tags and data to a new .tsv file.
        The new .tsv files shall be named as such:

            <corpus_name>_annotated.tsv

        where <corpus_name> is replaced with the name of the corpus file
            without the file extension.

        Args:
            corpus (str): The path to the corpus file
        """
        if VERBOSE:
            print("Annotating...")

        outfile = corpus.split(".")[0] + "_annotated.tsv"

        with open(outfile, mode='w', encoding='utf-8') as output:
            text = open(corpus).read()

            test_words = split_words(text)

            tagged_rows = self.tag_list(test_words)

            output.write("Token\tLanguage\tNamed Entity"
                         "\tEng-NGram Prob\tSpn-NGram Prob"
                         "\tHMM Prob\tTotal Prob\n")

            for row in tagged_rows:
                csv_row = '\t'.join(str(s) for s in row)
                if VERBOSE:
                    print(csv_row)
                output.write(csv_row + "\n")

        if VERBOSE:
            print("Annotation file written")


    def evaluate(self, gold_standard):
        """Evaluates the system, comparing system output to the gold standard's
            tags. The final file will be:

        <gold_standard>_evaluation.tsv

        Args:
            gold_standard (str): The path to the gold standard
        """
        if VERBOSE:
            print("Evaluating Performance...")

        outfile = gold_standard.split(".")[0] + "_evaluation.tsv"

        with open(outfile, mode='w', encoding='utf-8') as output:
            # create list of text and tags
            lines = open(gold_standard, 'r', encoding='utf-8').readlines()
            text, gold_tags = [], []

            # Get tokens and gold tags from gold standard
            for x in lines:
                columns = x.split(CONFIGS["gold_delimiter"])
                text.append(columns[-2].strip())
                gold_tags.append(columns[-1].strip())

            # Tag the text based on the provided models
            annotated_output = self.tag_list(text)
            _, lang_tags, NE_tags, _, _, _, _  = map(list, zip(*annotated_output))

            # Reset counters to 0, prepare for checking with the gold_standard
            langCorrect = langTotal = NECorrect = NETotal = 0
            evals = []

            # Compare gold standard and model tags
            for lang, NE, gold in zip(lang_tags, NE_tags, gold_tags):
                # Evaluate language tags
                if gold in CONFIGS["lang_set"]:
                    langTotal += 1
                    if gold == lang:
                        langCorrect += 1
                        evals.append("Correct")
                    else:
                        evals.append("Incorrect")

                # Evaluate NE tags
                elif gold == CONFIGS["ne_tag"]:
                    NETotal += 1
                    if NE != 'O':
                        NECorrect += 1
                        evals.append("Correct")
                    else:
                        evals.append("Incorrect")

                # Don't evaluate punctuation or number
                else:
                    evals.append("NA")

            # Write the final results to file
            output.write("Language Accuracy: {}\n".format(
                langCorrect / float(langTotal)))
            output.write("NE Accuracy: {}\n".format(
                NECorrect / float(NETotal)))

            output.write("Token\tGold Standard\tTagged Language"
                         "\tNamed Entity\tEvaluation\n")

            for all_columns in zip(text, gold_tags, lang_tags, NE_tags, evals):
                output.write("\t".join(all_columns) + "\n")

            if VERBOSE:
                print("Evaluation file written")


# eval.py [gold_standard] test_corpus
def main(argc, argv):
    """Main prep work and evaluation. Process:
    1. Get corpora
    2. Create NGram models
    3. Create Code-Switch Model
    4. Build Markov model with Expectation Minimization
    5. Annotate
    6. Optionally evaluate performance on gold standard
    """
    n = CONFIGS["ngram"]
    tagset = list(CONFIGS["lang_set"])

    # Process training corpora
    lang1_data = split_words(open(CONFIGS["lang1_train"], mode="r").read())
    lang2_data = split_words(open(CONFIGS["lang2_train"], mode="r").read())

    # Create language model of training corproa
    lang1_model = CNGram(tagset[0], lang1_data, n)
    lang2_model = CNGram(tagset[1], lang2_data, n)
    cs_model = CodeSModel([lang1_model, lang2_model])

    # Extract tags from gold standard
    gold_standard = open(CONFIGS["gold_path"], mode="r")
    gold_delimiter = CONFIGS["gold_delimiter"]
    gold_tags = [x.split(gold_delimiter)[-1].strip() for x in gold_standard.readlines()]

    # Convert all tags to either lang1 or lang2 and remove others
    gold_tags = [tagset[0] if x in CONFIGS["lang1_other"] else x for x in gold_tags]
    gold_tags = [tagset[1] if x in CONFIGS["lang2_other"] else x for x in gold_tags]
    gold_tags = [x for x in gold_tags if x in tagset]

    # Compute prior based on gold standard
    transitions = get_transi_matrix(gold_tags, tagset)

    # Create evaluator for input corpus, annotate, and evaluate
    eval = Evaluator(cs_model, transitions, tagset)
    eval.annotate(CONFIGS["infile"])
    eval.evaluate(CONFIGS["gold_path"])


def parse_config():
    """
    Parse parameters from config file.
    """
    config = ConfigParser()
    config_dict = {}

    # Must have config file
    if not os.path.isfile("config.ini"):
        print("Config file not found!")
        sys.exit(-1)

    config.read("config.ini")

    default = config["DEFAULT"]
    gold = config["GOLD"]
    advanced = config["ADVANCED"]

    CONFIGS["lang_set"] = set(default["lang_set"].split(","))
    CONFIGS["ngram"] = default.getint("ngram")
    CONFIGS["tokenize"] = default.getboolean("tokenize")
    CONFIGS["header"] = default.getboolean("header")
    CONFIGS["verbose"] = default.getboolean("verbose")

    if gold["lang1_other"]:
        CONFIGS["lang1_other"] = set(gold["lang1_other"].split(","))

    if gold["lang2_other"]:
        CONFIGS["lang2_other"] = set(gold["lang2_other"].split(","))

    if gold["other_tags"]:
        CONFIGS["other_tags"] = set(gold["other_tags"].split(","))

    CONFIGS["ner_chunk_size"] = advanced.getint("ner_chunk_size")

    # Put remaining options into global dict
    for section in config:
        for value in config[section]:
            if value not in CONFIGS:
                CONFIGS[value] = config[section][value]


def parse_args():
    global VERBOSE, HEADER, TOKENIZE

    # Optionally override some config options with arguments
    parser = argparse.ArgumentParser(
            description="Tag a mixed-language text by language")

    # Some optional arguments
    parser.add_argument(
            "--ngram",
            metavar="ngram",
            type=int,
            default=5,
            help="size of character ngrams (Default: 5)")
    parser.add_argument(
            "--tokenize",
            action="store_true",
            help="tokenize flag (Default: False)")
    parser.add_argument(
            "--header",
            action="store_true",
            help="header flag (Default: False)")
    parser.add_argument(
            "--gold-delimiter",
            type=str,
            default="\t",
            help="delimiter for gold standard file (Default: tab)")
    parser.add_argument(
            "-v", "--verbose",
            action="store_true",
            help="verbose flag (Default: False)")

    # Some positional arguments
    parser.add_argument(
            "infile",
            nargs="?",
            type=str,
            help="corpus file (Default: stdin)")

    args = parser.parse_args()

    if args.verbose:
        VERBOSE = True

    if args.header:
        HEADER = True

    if args.tokenize:
        TOKENIZE = True

    # Update global options dict
    CONFIGS.update(vars(args))


if __name__ == "__main__":
    """
    Parse options in config file and command-line arguments and then run main
    function.

    Note: This code only runs when the script is executed on the command line.
    """
    parse_config()
    parse_args()

    if VERBOSE:
        print(CONFIGS)

    main(len(sys.argv), sys.argv)
