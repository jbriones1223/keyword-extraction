import argparse
import csv
import numpy as np

# Given the name of a CSV file, return a list of the rows in the file.
def get_inputs(filename):
    f = open(filename, 'rb')
    reader = csv.reader(f)
    inputs = []
    for row in reader:
        inputs.append(row)
    return inputs

# Given a list of lists of rankings, return a set of all phrases appearing at
# least once.
def get_vocab(inputs):
    vocab = set()
    for row in inputs:
        vocab.update(frozenset(row))

    return frozenset(vocab)

# Given inputs and a set of vocabulary, return a dictionary of each word's
# positions in the inputs.
def get_positions(inputs, vocab):
    max_rank = max(inputs, key = lambda x: len(x))

    positions = dict([(word, []) for word in vocab])
    for row in inputs:
        for word in positions:
            if word in row:
                positions[word].append(float(max_rank - row.index(word)))
            else:
                positions[word].append(0.)

    return positions

# Given a dictionary of each word's positions, return a dictionary mapping each
# word to a list of data about that word.
# Current form of the list:
#       word : [min, max, range, variance]
def get_data(positions):
    data = {}
    for word in positions:
        p = positions[word]
        data[word] = [min(p), max(p), max(p) - min(p), np.var(p, ddof=1)]

    return data

if __name__ == "__main__":
    # Command-line argument handling
    parser = argparse.ArgumentParser()
    parser.add_argument("readfile", help="file to read")
    parser.add_argument("--num_kw", type=int, help="number of results to return. default behavior returns all results")
    parser.add_argument("-s", "--savefile", help="destination to save results. default behavior only prints")
    style = parser.add_mutually_exclusive_group()
    style.add_argument("--mini", action="store_true", help="sort results by minimum rank")
    style.add_argument("--maxi", action="store_true", help="sort results by maximum rank")
    style.add_argument("--span", action="store_true", help="sort results by difference between min and max rank")
    style.add_argument("--variance", action="store_true", help="sort results by variance. this is the default behavior")
    args = parser.parse_args()

    # list of lists, each containing some number of top results, best first.
    inputs = get_inputs(args.readfile)

    # set of vocab occurring in any of the lists from inputs
    vocab = get_vocab(inputs)

    # get a dictionary of lists containing each word's positions in the rankings
    positions = get_positions(inputs, vocab)

    # get data from the positions
    data = get_data(positions)

    # NOTE: default statistic is variance
    idx = 3
    if args.span:
        idx = 2
        print 'span'
    elif args.maxi:
        idx = 1
        print 'maximum'
    elif args.mini:
        idx = 0
        print 'minimum'

    final = [(word, data[word][idx]) for word in data]
    final.sort(key = lambda x: x[1])
    final.reverse()
    if args.num_kw:
        final = final[:args.num_kw]

    if not args.savefile:
        for pair in final:
            print pair[0] + ' : ' + str(pair[1])
