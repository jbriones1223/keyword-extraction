import codecs
import csv
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
import re
import math
import argparse

#==============================================================================#
# Globals
#==============================================================================#
NOUN = False
GRAMS = True
UNIGRAM = True
BIGRAM = True
TRIGRAM = True
SELECTIVITY = False
TFIDF = True

#==============================================================================#
# Function Definitions
#==============================================================================#
# Given the name of the CSV file, return a list of the tweets it contains
def get_tweets(file_name):
    read_file = codecs.open(file_name, 'rb')
    reader = csv.reader(read_file)
#   reader.next()
#   return map(str.lower, reader.next()[1:])
#   return [unicode(row[2], encoding='utf-8', errors='ignore') for row in reader][1:]
    tweets = []
    for row in reader:
        tweets.append(row[2].decode('utf-8').lower())
    return tweets

# Given a list of tweets, break each tweet into sentences and return a list of
# all sentences.
def get_sents(tweets):
    sents = []
    for tweet in tweets:
        sents += sent_tokenize(tweet)
    return sents

# Given a list of regular expressions to filter out and a list of sentences,
# remove all matching instances from the sentences. Return the result.
def re_filter(sentence_list, re_list):
    for i in range(len(sentence_list)):
        for rgx in re_list:
            sentence_list[i] = rgx.sub("", sentence_list[i])
    return sentence_list

# Given a list of sentences, and a list of stop-phrases, remove all phrases from
# the sentences. Longer phrases are removed first, because removing one word at
# a time will create issues with detecting full stop-phrases.
def remove_stops(sentences, stops):
    unistops = [stop for stop in stops if len(stop.split()) == 1]
    bistops = [stop for stop in stops if len(stop.split()) == 2]
    tristops = [stop for stop in stops if len(stop.split()) == 3]
    sorted_stops = [tristops, bistops, unistops]
    rgxs = []
    for stoplist in sorted_stops:
        for stop in stoplist:
            rgxs.append(re.compile('\\W' + '\\W+'.join(stop.split()) + '(?=\\W)'))
    for i in range(len(sentences)):
        for rgx in rgxs:
            sentences[i] = rgx.sub(' ',sentences[i])
    return sentences

# Given a list of tweets, return the set of all words encountered and a list of
# how many terms are in each tweet. Note that the list of tweets should have
# already gone through stop phrase removal. Additionally, note that the returned
# set is a frozenset (slightly faster).
def get_vocab(tweet_list):
    # Create list for counts of terms
    counts = []

    # Remove non-alphanumeric characters
    for i in range(len(tweet_list)):
        tweet_list[i] = re.sub("'", '', tweet_list[i])
        tweet_list[i] = re.sub('\\W', ' ', tweet_list[i])
        counts.append(len(tweet_list[i].split()))

    # Create a set to store the words found.
    vocab_set = set()

    # Loop through each tweet, updating the vocabulary set along the way.
    for tweet in tweet_list:
        vocab_set.update(frozenset(tweet.split()))

    return frozenset(vocab_set), counts

# Given a list of tweets and a vocabulary set, return a dictionary mapping words
# to their inverse document frequencies.
def get_idf(tweet_list, vocab_set):
    # Create the dictionary.
    idf = {}

    # For each word, count the number of documents it appears in.
    for word in vocab_set:
        idf[word] = float(sum([word in tweet for tweet in tweet_list]))

    N = len(tweet_list)

    # Compute the idf score for each word. I went with the standard formula, but
    # that can be changed:
    # https://en.wikipedia.org/wiki/Tf-idf#Inverse_document_frequency
    for word in idf:
        idf[word] = math.log(N / idf[word])

    return idf

# Given a list of tweets, a vocabulary set, and a list of how many terms are in 
# each tweet, compute each term frequency for each tweet, and return it as a a
# list of dictionaries.
def get_tf(tweet_list, vocab_set, term_counts):
    tf = []
    for i in range(len(tweet_list)):
        df = {}
        for word in vocab_set:
            df[word] = float(tweet_list[i].count(word))
        # This is the standard calculation, but there are other formulas:
        # https://en.wikipedia.org/wiki/Tf-idf#Term-frequency
        for word in df:
            df[word] /= term_counts[i]
        tf.append(df)

    return tf

# Given a list of term frequency dictionaries and an inverse document frequency
# dictionary, return a list of dictionaries representing each term's weight in
# each tweet.
def get_weights(tf, idf):
    return [dict([(word, idf[word] * tf_dict[word]) for word in tf_dict]) for tf_dict in tf]

# Given a list of term frequency dictionaries, return a dictionary containing
# each word's total weight.
def get_totals(weights):
    totals = {}
    for doc in weights:
        for word in doc:
            if word in totals:
                totals[word] += doc[word]
            else:
                totals[word] = doc[word]

    return totals

# Given a list of tweets and a desired number of results, return a list of the
# highest scoring vocabulary words, using TF-IDF
# FIXME: This function ```works as intended''', but the results are unhelpful.
#        Consider changing the scoring methods.
def tf_idf(tweet_list, num_kw):
    vocab_set, term_counts = get_vocab(tweet_list)
    idf = get_idf(tweet_list, vocab_set)
    tf = get_tf(tweet_list, vocab_set, term_counts)
    weights = get_weights(tf, idf)
    totals = get_totals(weights)
    totals = [(totals[word], word) for word in totals]
    totals.sort()
    totals.reverse()
    return totals[:num_kw]

#==============================================================================#
# File I/O
#==============================================================================#
# NOTE: text should be in unicode format. Depending on our sources, I'll look
#       into format conversions.

if __name__ == "__main__":
# Get the name of the CSV file containing the tweets
    parser = argparse.ArgumentParser()
    parser.add_argument('readfile', help='name of dataset to analyze')
    parser.add_argument('-n', help='number of results to return', type=int, default=25)
    parser.add_argument('--save', dest='savefile', help='file to save results')
    parser.add_argument('-uni', action='store_true', help='return unigram analysis')
    parser.add_argument('-bi', action='store_true', help='return bigram analysis')
    parser.add_argument('-tri', action='store_true', help='return trigram analysis')
    parser.add_argument('-noun', action='store_true', help='return noun frequency analysis')
    parser.add_argument('-sel', action='store_true', help='return selectivity analysis')
    parser.add_argument('-tfidf', action='store_true', help='return TF-IDF analysis')
    args = parser.parse_args()

# Decide between saving results and printing results
    write_handle = 0
    csv_writer = 0
    if args.savefile:
        write_handle = codecs.open(args.savefile, 'wb', 'utf-8')
        csv_writer = csv.writer(write_handle)

# Get the tweets
    tweets = get_tweets(args.readfile)

# remove hashtags, URLs, and mentions here
#               Note: these can easily be condensed into one regular expression,
#               but for now I will leave them separate for easy testing purposes
    mention_form = re.compile('@\\S+')
    url_form = re.compile('http\\S+')
    hashtag_form = re.compile('#\\S+')
# https://stackoverflow.com/a/33417311
# TODO: this doesn't work right now, I'm not sure why. will do some digging.
# emoji_form = re.compile("["
#         u"\U0001F600-\U0001F64F"  # emoticons
#         u"\U0001F300-\U0001F5FF"  # symbols & pictographs
#         u"\U0001F680-\U0001F6FF"  # transport & map symbols
#         u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
#                            "]+", flags=re.UNICODE | re.VERBOSE)

    tweets = re_filter(tweets, [mention_form, url_form, hashtag_form])

# Introduce default stopwords
    stop_words = set(stopwords.words('english'))

# Let user define some of their own stop words
    print("Enter any additional stop words, one at a time. Press enter \
 when you are finished. Limit to n-grams with n at most 3. ")
    while (True):
        new_word = raw_input("Add a stop word: ")
        if not new_word:
            break
        stop_words.add(new_word.lower())
    print 

    fraud_words = ["election fraud", "election manipulation", "illegal voters",
                    "illegal votes", "dead voters", "noncitizen voting",
                    "noncitizen votes", "illegal voting", "illegal vote",
                    "illegal ballot", "illegal ballots", "dirty voter rolls",
                    "vote illegally", "voting illegally", "voter intimidation",
                    "voter suppression", "rigged election", "vote rigging",
                    "voter fraud", "voting fraud", "vote buying", "vote flipping",
                    "flipped votes", "voter coercion", "ballot stuffing",
                    "ballot box stuffing", "ballot destruction",
                    "voting machine tampering", "rigged voting machines",
                    "voter impersonation", "election integrity", "election rigging",
                    "duplicate voting", "duplicate vote", "ineligible voting",
                    "ineligible vote"]
    electionDay_words = ["provisional ballot", "voting machine", "ballot"]
    pollingPlaces_words = ["polling place line", "precinct line", "pollworker",
                            "poll worker"]
    remoteVoting_words = ["absentee ballot", "mail ballot", "vote by mail",
                            "voting by mail", "early voting"]
    voterID_words = ["voter identification", "voting identification", "voter id"]
    track_words = fraud_words + electionDay_words + pollingPlaces_words + \
                    remoteVoting_words + voterID_words

# list of all phrases to remove
    stop_words = set.union(stop_words, set(track_words))

# list of POS tags to remove
    removal = set(['.', 'NUM'])

# remove stop phrases
    print 'Removing stop phrases'
    tweets = remove_stops(tweets, stop_words)

    # Parse the tweets into sentences
    sents = get_sents(tweets)

    # Parse each sentence into words.
    words = [word_tokenize(s) for s in sents]

    # Create list for POS-tagged words
    words_tagged = []

    for i in range(len(words)):
        # Perform POS tagging
        tagged = nltk.pos_tag(words[i], tagset='universal')
        # Remove punctuation and numbers
        # TODO: fix punctuation. it seems like the unicode apostrophe gets through
        #       the filter, even though the ' character is successfully removed.
        words[i] = [w[0] for w in tagged if not w[1] in removal]
        words_tagged += [w for w in tagged if not w[1] in removal]

    # Sort the text into n-grams, one sentence at a time.
    ngrams = []
    for sentence in words:
        # unigrams, bigrams, and trigrams - change max_len to get more or less.
        ngrams += list(nltk.everygrams(sentence, max_len = 3))

    # ngrams includes all types of n-grams. get lists for each type as well.
    unigrams = [n for n in ngrams if len(n) == 1]
    bigrams = [n for n in ngrams if len(n) == 2 and not n[0][0] + " " + n[1][0] in track_words]
    trigrams = [n for n in ngrams if len(n) == 3 and not n[0][0] + " " + n[1][0] + " " + n[2][0] in track_words]

    # Get the frequency distribution of each
    unigrams_fd = nltk.FreqDist(unigrams)
    bigrams_fd = nltk.FreqDist(bigrams)
    trigrams_fd = nltk.FreqDist(trigrams)

# set the number of keywords to extract from each method
    num_kw = args.n

# Get the most common of each:
    if args.uni:
        if not args.savefile:
            print str(num_kw) + " most common unigrams:"
            # print len(unigrams_fd)
            for w in unigrams_fd.most_common(num_kw):
                # w has the form ((5,),5)
                print w[0][0]
        else:
            csv_writer.writerow([str(num_kw) + " most common unigrams"] + [w[0][0]
            for w in unigrams_fd.most_common(num_kw)])

    if args.noun:
        nouns = [w[0] for w in words_tagged if w[1] == 'NOUN']
        nouns_fd = nltk.FreqDist(nouns)
        if not args.savefile:
            print "\n" + str(num_kw) + " most common nouns:"
            # print len(nouns_fd)
            for w in nouns_fd.most_common(num_kw):
                # w has form (5, 5)
                print w[0]
        else:
            csv_writer.writerow([str(num_kw) + " most common nouns"] + [w[0]
            for w in nouns_fd.most_common(num_kw)])

    if args.bi:
        if not args.savefile:
            print "\n" + str(num_kw) + " most common bigrams:"
            # print len(bigrams_fd)
            for w in bigrams_fd.most_common(num_kw):
                # w has for ((5,5),5)
                print w[0][0] + " " + w[0][1]
        else:
            csv_writer.writerow([str(num_kw) + " most common bigrams"] + [w[0][0] + " " + w[0][1]
            for w in bigrams_fd.most_common(num_kw)])

    if args.tri:
        if not args.savefile:
            print "\n" + str(num_kw) + " most common trigrams:"
            # print len(trigrams_fd)
            for w in trigrams_fd.most_common(num_kw):
                print w[0][0] + " " + w[0][1] + " " + w[0][2]
        else:
            csv_writer.writerow([str(num_kw) + " most common trigrams"] + [w[0][0] + " " + w[0][1] + " " + w[0][2]
            for w in trigrams_fd.most_common(num_kw)])

    if args.sel:
        # Now, store the entries from bigrams_fd to compute selectivity results.
        degree = {}
        strength = {}

        # current form: co-occurrence, weighted and undirected
        most_common = bigrams_fd.most_common()
        for ((a, b), c) in most_common:
            if not a in degree.keys():
                degree[a] = 0
            if not a in strength.keys():
                strength[a] = 0
            if not b in degree.keys():
                degree[b] = 0
            if not b in strength.keys():
                strength[b] = 0
            degree[a] += 1
            degree[b] += 1
            strength[a] += c
            strength[b] += c

        # calculate selctivity for each word
        selectivity = []
        for w in degree.keys():
            selectivity.append((float(strength[w]) / degree[w], w))

        # Result extraction:

        selectivity.sort()
        selectivity.reverse()

        if not args.savefile:
            print "\n" + str(num_kw) + " best selectivity scores:"
            # print len(selectivity)
            for (n, s) in selectivity[:num_kw]:
                print s + " : " + str(n)
        else:
            csv_writer.writerow([str(num_kw) + " best selectivity scores"] + [s + " : " + str(n)
            for (n, s) in selectivity[:num_kw]])

    if args.tfidf:
        print 'Calculating top ' + str(num_kw) + ' best TF-IDF scores:\n'
        for (weight, word) in tf_idf(tweets, num_kw):
            print word + ' : ' + str(weight)
