import argparse
import csv
from grabber3 import get_tweets



# Format the tweets so they can easily be searched. For now, no extra formatting
# is done.
def clean(tweets_raw):
    return tweets_raw

# Check that the tweet contains the phrase
def contains(tweet, phrase):
    return all([' ' + w + ' ' in tweet for w in phrase.split()])

# Get the relevant context for the phrase.
def extract_context(tweet, phrase):
    return tweet

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("readfile", help='location of tweets')
    parser.add_argument("phrase", nargs='+', help='keywords to find context of')
    args = parser.parse_args()

    num_contexts = 5

    tweets_raw = get_tweets(args.readfile)
    phrases = args.phrase

    contexts = dict([(phrase, {}) for phrase in phrases])

    tweets_nice = clean(tweets_raw)

    for i in range(len(tweets_nice)):
        for phrase in contexts:
            if contains(tweets_nice[i], phrase):
                c = extract_context(tweets_nice[i], phrase)
                if c in contexts[phrase]:
                    contexts[phrase][c] += 1
                else:
                    contexts[phrase][c] = 1

    for phrase in contexts:
        d = [(c, contexts[phrase][c]) for c in contexts[phrase]]
        d.sort(key = lambda x: x[1])
        d.reverse()
        print('Top ' + str(num_contexts) + ' contexts for ' + phrase + ':\n')
        for c in d[:num_contexts]: print(c[0] + '\n\n\n')
        print()

