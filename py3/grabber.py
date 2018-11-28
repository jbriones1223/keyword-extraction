import codecs
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
import re

#==============================================================================#
# File I/O
#==============================================================================#
# NOTE: text should be in unicode format. Depending on our sources, I'll look
#       into format conversions.

file_name = input("Enter file name: ")
if not file_name: file_name = "twitter_as_data.txt"

file_in = codecs.open(file_name, "r", encoding="utf-8")
text = file_in.read()
sents = sent_tokenize(text)

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
for i in range(len(sents)):
    sents[i] = mention_form.sub("", sents[i])
    sents[i] = hashtag_form.sub("", sents[i])
    sents[i] = url_form.sub("", sents[i])
#     sents[i] = emoji_form.sub("", sents[i])

words = [word_tokenize(s) for s in sents]

# Introduce default stopwords
stop_words = set(stopwords.words('english'))

# Let user define some of their own stop words
print("Enter any additional stop words, one at a time. Press enter when you \
are finished.")
while (True):
    new_word = input("Add a stop word: ")
    if not new_word:
        break
    stop_words.add(new_word)
print() 

removal = set(['.', 'NUM'])

for i in range(len(words)):
    # Switch sentence to lower case
    words[i] = [w.lower() for w in words[i]]
    # Perform POS tagging
    words[i] = nltk.pos_tag(words[i], tagset='universal')
    # Remove stop words, punctuation, and numbers
    # TODO: fix punctuation. it seems like the unicode apostrophe gets through
    #       the filter, even though the ' character is successfully removed.
    words[i] = [w for w in words[i] if not w[0] in stop_words
                and not w[1] in removal]

# Sort the text into n-grams, one sentence at a time.
ngrams = []
for sentence in words:
    # unigrams, bigrams, and trigrams - change max_len to get more or less.
    ngrams += list(nltk.everygrams(sentence, max_len = 3))

# ngrams includes all types of n-grams. get lists for each type as well.
unigrams = [n for n in ngrams if len(n) == 1]
bigrams = [n for n in ngrams if len(n) == 2]
trigrams = [n for n in ngrams if len(n) == 3]

# Get the frequency distribution of each
ngrams_fd = nltk.FreqDist(ngrams)
unigrams_fd = nltk.FreqDist(unigrams)
bigrams_fd = nltk.FreqDist(bigrams)
trigrams_fd = nltk.FreqDist(trigrams)

# set the number of keywords to extract from each method
num_kw = 10

# Get the most common of each:
print(str(num_kw) + " most common unigrams:")
for w in unigrams_fd.most_common(num_kw):
    print(w[0][0][0])

print("\n" + str(num_kw) + " most common nouns:")
nouns = [w for w in unigrams if w[0][1] == 'NOUN']
nouns_fd = nltk.FreqDist(nouns)
for w in nouns_fd.most_common(num_kw):
    print(w[0][0][0])

print("\n" + str(num_kw) + " most common bigrams:")
for w in bigrams_fd.most_common(num_kw):
    print(w[0][0][0] + " " + w[0][1][0])

print("\n" + str(num_kw) + " most common trigrams:")
for w in trigrams_fd.most_common(num_kw):
    print(w[0][0][0] + " " + w[0][1][0] + " " + w[0][2][0])

# Now, store the entries from bigrams_fd to compute selectivity results.
degree = {}
strength = {}

# current form: co-occurrence, weighted and undirected
most_common = bigrams_fd.most_common()
for ((a, b), c) in most_common:
    if not a in list(degree.keys()):
        degree[a] = 0
    if not a in list(strength.keys()):
        strength[a] = 0
    if not b in list(degree.keys()):
        degree[b] = 0
    if not b in list(strength.keys()):
        strength[b] = 0
    degree[a] += 1
    degree[b] += 1
    strength[a] += c
    strength[b] += c

# calculate selctivity for each word
selectivity = []
for w in list(degree.keys()):
    selectivity.append((float(strength[w]) / degree[w], w))

# Result extraction:

print("\n" + str(num_kw) + " best selectivity scores:")
selectivity.sort()
selectivity.reverse()
for (n, s) in selectivity[:num_kw]:
    print(s[0] + " : " + str(n))

