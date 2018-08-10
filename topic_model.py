import csv
import codecs
import argparse
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import string
import gensim
from gensim import corpora
import warnings



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
        tweets.append(' ' + row[2].decode('utf-8').lower())
    return tweets

def clean(tweet, stop, exclude, lemma):
    stop_free = " ".join([i for i in tweet.lower().split() if i not in stop])
    punc_free = ''.join([ch for ch in stop_free if ch not in exclude])
    normalized = " ".join([lemma.lemmatize(word) for word in punc_free.split()])

    return normalized

if __name__ == "__main__":
    warnings.filterwarnings("ignore")

    parser = argparse.ArgumentParser()
    parser.add_argument('readfile', help='location of data to analyze')
    parser.add_argument('-n', help='number of topics to return', type=int, default=25)
    args = parser.parse_args()

    tweets = get_tweets(args.readfile)

    stop = set(stopwords.words('english'))
    exclude = set(string.punctuation)
    lemma = WordNetLemmatizer()

    tweets_clean = [clean(tweet, stop, exclude, lemma).split() for tweet in tweets]

    dictionary = corpora.Dictionary(tweets_clean)

    doc_term_matrix = [dictionary.doc2bow(tweet) for tweet in tweets_clean]

    Lda = gensim.models.ldamodel.LdaModel

    ldamodel = Lda(doc_term_matrix, num_topics = args.n, id2word = dictionary, passes = 1)

    print(ldamodel.print_topics(num_topics=args.n, num_words=args.n))
