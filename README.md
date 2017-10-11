# Code-Switching Language Tagger

## General Description
``python-tagger`` is a language identification tool written in Python based on [similar code](https://github.com/Bilingual-Annotation-Task-Force/codeswitch-annotation) written in Scala. Note that this code has only been tested on the language pairs English-Spanish and English-French.

## Requirements
To run the script, the following files are required:
- training corpora for each language in UTF-8,
- the latest version of the Stanford Named Entity Recognizer from [here](https://nlp.stanford.edu/software/CRF-NER.html#Download),
- a gold standard for computing priors and evaluation in tab-separated format.

Additionally, running it requires Python3 and NLTK.
The script assumes that the gold standard is already tokenized with the language tags in the rightmost column.
Note that as of now, the only classifiers supported for Named Entity Recognition are for English and Spanish.

## Usage
    eval.py [options] infile

```
usage: evaluator.py [-h] [--ngram ngram] [--tokenize] [--header]
                    [--gold-delimiter GOLD_DELIMITER] [-v]
                    [infile]

Tag a mixed-language text by language

positional arguments:
  infile                corpus file (Default: stdin)

optional arguments:
  -h, --help            show this help message and exit
  --ngram ngram         size of character ngrams (Default: 5)
  --tokenize            tokenize flag (Default: False)
  --header              header flag (Default: False)
  --gold-delimiter GOLD_DELIMITER
                        delimiter for gold standard file (Default: tab)
  -v, --verbose         verbose flag (Default: False)
  ```
  
  Further options in `config.ini` file:
- [DEFAULT]
  - LANG_SET = Comma-separated list of language tags corresponding to language tags in gold standard
  - NGRAM = Size of ngram model
  - TOKENIZE = Tokenize test corpus (unused)
  - HEADER = Presence of Header row included in test corpus
  - VERBOSE = Print diagnostic information

- [TRAIN_PATHS]
  - LANG1_TRAIN = Path to training data for Language 1
  - LANG2_TRAIN = Path to training data for Language 2

- [CLASS_PATHS]
  - CLASS_JAR = Path to Stanford NER .jar file
  - LANG1_CLASS = Path to NER classifier for Language 1
  - LANG2_CLASS = Path to NER classifier for Language 2

- [GOLD]
  - GOLD_PATH = Path to gold standard
  - GOLD_DELIMITER = Quoted delimeter for gold standard
  - LANG1_OTHER = Other tags in gold standard corresponding to Language 1
  - LANG2_OTHER = Other tags in gold standard corresponding to Language 2
  - NE_TAG = Tag in gold standard corresponding to Named Entity
  - OTHER_TAGS = Unwanted tags in gold standard (unused)

- [ADVANCED]
  - NER_CHUNK_SIZE = Token batch size for calls to Named Entity Recognizer

 ### TODO
- [x] Translate from Scala
- [ ] Update and document code
- [ ] Write test code
- [ ] Convert to Python package
