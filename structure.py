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
import string

from grabber3 import get_tweets, re_filter

#==============================================================================#
# Function Definitions
#==============================================================================#

# Clean a tweet. Keep in mind that this may be called on either a full tweet or
# a single sentence, depending on the value of <sent_parse>. However, sentence
# parsing will have already happened, so don't worry about messing that up (i.e.
# you CAN get by with removing periods.
def clean(tweet):
    # Split on non-alphanumeric characters, and remove any empty strings
    return [e for e in re.split('\W+', tweet.lower()) if e]

# Perform preprocessing on the list of tweets, returning a list of lists of
# words.
def process_tweets(tweets):
    return [clean(tweet) for tweet in tweets]

# Given a tweet and a list of search phrases, find the index of each occurring
# search phrase. Return this data as a dictionary mapping lengths of search
# terms to lists of indices.
def find_phrases(tweet, search):
    locations = {}
    l = len(tweet)
    if not l: return {}
    for term in search:
        if not term: continue
        # split search term into separate words, and store the length.
        wl = term.lower().split()
        tl = len(wl)
        i = -1 * tl
        locs = []
        try:
            while 1:
                # find the index. this is where a ValueError may be raised.
                try:
                    idx = tweet[i + tl:].index(wl[0])
                except IndexError: # NOTE: used for error checking. will remove later
                    print 'tweet = ' + str(tweet)
                    print 'wl = ' + str(wl)
                    print 'tl = ' + str(tl)
                    print 'i = ' + str(i)
                    raise IndexError
                # check if the word actually occurred:
                if i + 2 * tl + idx <= l and \
                    all([tweet[i + tl + idx + j] == wl[j] for j in range(tl)]):
                    # if it occurred, mark the position and continue searching
                    i += idx + tl
                    locs.append(i)
                else:
                    i += idx + 1
        except ValueError:
            if locs:
                if not tl in locations:
                    locations[tl] = []
                locations[tl] += locs
            continue
    return locations

# Given a tweet and a dictionary mapping search term lengths to search term
# locations, return a dictionary mapping contexts to the number of times they
# appeared in the tweet. Each context is represented as list of words.
def find_contexts(tweet, locs, size):
    l = len(tweet)
    if not l: return {}
    contexts = {}
    for length in locs:
        for loc in locs[length]:
            c1 = tweet[max(0, loc - size) : loc]
            c2 = tweet[loc + length : loc + length + size]
            # c is the context
            # TODO: make sure tweets can't contain <|>
            c = ' '.join(c1 + ['<|>'] + c2)
            if not c in contexts:
                contexts[c] = 0
            contexts[c] += 1
    return contexts

def get_scores(processed_tweets, search_phrases, contexts, min_size, max_size):
    scores = {}
    if min_size < 1 or max_size < 1:
        raise ValueError('invalid min_size or max_size value')
    for tweet in processed_tweets: # for each tweet (each is a list of words)
        for c in contexts: # for each context (each is a single string)
            pt = c.split()
            l = len(pt) - 1
            if l < 1:
                continue
            try:
                b = pt.index('<|>')
            except ValueError:
                continue
            first = pt[:b]
            last = pt[b+1:]
            if not first or not last: continue
            starts = find_phrases(tweet, [ ' '.join(first) ])
            ends = find_phrases(tweet, [' '.join(last)])
            if starts and ends:
                starts = starts[b]
                starts = [s + b for s in starts]
                ends = ends[l-b]
            else:
                continue
            for s in starts:
                for i in range(min_size, max_size + 1):
                    if s + i in ends:
                        p = ' '.join(tweet[s : s + i])
                        if p not in scores:
                            scores[p] = 0
                        # TODO: change to scoring by frequency of context?
                        scores[p] += 1
    return scores

def filter_results(scores):
    return scores

# The simplest method, more to come. This first finds which phrases frequently
# occur around keywords, and stores them. Next, it finds which other words also
# appear in those contexts. During this step, the algorithm creates and updates
# a dictionary of every word which has a nonzero score. This dictionary is then
# passed to a filter to select the words with the best scores, which are then
# returned.
def find_keywords_basic(processed_tweets, search_phrases, num_kw, size, min_size, max_size):
    if size < 1:
        raise ValueError('size of context must be at least 1')
    contexts = {}
    i = 0
    for pt in processed_tweets:
        # Get a dictionary storing locations of all key phrases. Maps each
        # distinct keyphrase length to a list of indices where a first term
        # occurs.
        locations = find_phrases(pt, search_phrases)
        # find all contexts present and return a dict mapping them to their counts
        context = find_contexts(pt, locations, size)
        # update the overall dictionary
        for c in context:
            if c in contexts:
                contexts[c] += context[c]
            else:
                contexts[c] = context[c]
        i += 1
    # get a dictionary mapping phrases to scores
    scores = get_scores(processed_tweets, search_phrases, contexts, min_size, max_size)
    scores = filter_results(scores)
    sc_tup = [(keyphrase, scores[keyphrase]) for keyphrase in scores]
    sc_tup.sort(key=lambda x: -1 * x[1])
    return sc_tup[:num_kw]

# Given a list of tweets, parse each tweet into sentences, and return the list
# of all sentences. Note that this list is 1D, as opposed to a 2D list of lists.
def parse_sents(tweets):
    sent_list = []
    for tweet in tweets:
        sent_list += sent_tokenize(tweet)
    return sent_list

#==============================================================================#
# Global Constants
#==============================================================================#

# Whether or not to worry about structures that span multiple sentences. I think
# this could really be argued in either direction, but for now I'll leave it on,
# meaning we only consider the contents of one sentence at a time.
# Note that this will slightly affect preprocessing; having this off will
# result in a list of tweets, and having this on will result in a list of
# sentences.
sent_parse = False

#==============================================================================#
# Main
#==============================================================================#
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('readfile', help='name of dataset to analyze')
    parser.add_argument('-n', help='number of results to return', type=int, default=25)
    parser.add_argument('--save', dest='savefile', help='file to save results')
    parser.add_argument('-r', action='store_true', help='remove stop words for the algorithm')
    parser.add_argument('-c', action='store_true', help='complicate things. use this option to see whatever algorithm is in progress. not currently implemented')
    parser.add_argument('-size', help='size of context', type=int, default=2)
    parser.add_argument('-min', help='min size of results', type=int, default=1)
    parser.add_argument('-max', help='max size of results', type=int, default=3)
    args = parser.parse_args()

    tweets = get_tweets(args.readfile)

    # TODO: write a way to pull keywords from a csv file or from the command line
    search_phrases = ['provisional ballot', 'voting machine', 'ballot']

    if sent_parse:
        print 'parsing tweets'
        tweets = parse_sents(tweets)

    tweets = re_filter(tweets)

    processed = process_tweets(tweets)

    if args.c:
        print "I haven't done this yet. But the next idea will involve " + \
        "computing similarity scores, as opposed to requiring exact matches"
        raise SystemExit
    else:
        results = find_keywords_basic(processed, search_phrases, args.n, args.size, args.min, args.max)

        print results
