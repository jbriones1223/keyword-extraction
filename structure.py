import codecs
import csv
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
import re
import math
import argparse
from nltk.stem.snowball import SnowballStemmer
import sys

from grabber3 import get_tweets, re_filter

#==============================================================================#
# Function Definitions
#==============================================================================#
# Perform preprocessing on the list of tweets. <break_sents> is treated as a
# boolean value; tweets will be treated as a single sentence if <break_sents> is
# True, and tweets will go through a sentence parser. In either case, the tweets
# are returned as a list of lists of words.
def process_tweets(tweets, break_sents):
    return 0

# The simplest method, more to come. This first finds which phrases frequently
# occur around keywords, and stores them. Next, it finds which other words also
# appear in those contexts. During this step, the algorithm creates and updates
# a dictionary of every word which has a nonzero score. This dictionary is then
# passed to a filter to select the words with the best scores, which are then
# returned.
def find_keywords_basic(processed_tweets, search_phrases):
    return 0

#==============================================================================#
# Global Constants
#==============================================================================#

# Whether or not to worry about structures that span multiple sentences. I think
# this could really be argued in either direction, but for now I'll leave it on,
# meaning we only consider the contents of one sentence at a time.
# Note that this will slightly affect preprocessing; having this off will
# result in a list of tweets, and having this on will result in a list of
# sentences.
sent_parse = True

#==============================================================================#
# Main
#==============================================================================#
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('readfile', help='name of dataset to analyze')
    parser.add_argument('-n', help='number of results to return', type=int, default=25)
    parser.add_argument('--save', dest='savefile', help='file to save results')
    parser.add_argument('-r', action='store_true', help='remove stop words for the algorithm')
    parser.add_argument('-c', action='store_true', help='complicate things. use this option to see whatever algorithm is in progress.'
    args = parser.parse_args()

    tweets = get_tweets(args.readfile)

    # TODO: write a way to pull keywords from a csv file or from the command line
    search_phrases = ['placeholder']

    tweets = re_filter(tweets)

    processed = process_tweets(tweets, break_sents)

    if args.r:
        print "I haven't done this yet. But the next idea will involve " + \
        "computing similarity scores, as opposed to requiring exact matches"
        raise SystemExit
    else:
        results = find_keywords_basic(tweets)
