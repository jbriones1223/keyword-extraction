import codecs
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords



# Do precomputing on the text.
# TODO: consider the following:
#           how should we handle punctuation?

#==============================================================================#
# File I/O
#==============================================================================#
# Note: the input file should just be a plaintext. Possibly use a different
#       encoding.
file_in = codecs.open("twitter_as_data.txt", "r", encoding="utf-8")
text = file_in.read()
sents = sent_tokenize(text)
words = [word_tokenize(s) for s in sents]


# remove stopwords here
stop_words = set(stopwords.words('english'))

for i in range(len(words)):
    words[i] = [w for w in words[i] if not w in stop_words]

# Sort the text into bigrams, one sentence at a time.
bgrms = []
for sentence in words:
    bgrms += list(nltk.bigrams(sentence))

# Get the frequency distribution of the bigrams
bg_fd = nltk.FreqDist(bgrms)

# Now, store the entries from bg_fd.
degree = {}
strength = {}

# current form: co-occurrence, weighted and undirected
most_common = bg_fd.most_common()
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

# num_kw = ___          (number of keywords to select)
# selectivity.sort()
# selectivity.reverse()
# results = [____ for (n, s) in selectivity[:num_kw]]

#==============================================================================#
# POS tagging
#==============================================================================#

text_in = []
for sent in words:
    text_in += sent

tagged_text = nltk.pos_tag(text_in, tagset='universal')
# Only save nouns for now. This may change.
save_words = ["NOUN"]
tagged_text = [t for (t, u) in tagged_text if u in save_words]
tagged_fd = nltk.FreqDist(tagged_text)

# Result extraction:

# num_kw = ___          (number of keywords to select)
# results = [____ for (w, n) in tagged_fd.most_common(num_kw)]
