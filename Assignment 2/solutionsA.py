import math
import nltk
import time

# Constants to be used by you when you fill the functions
START_SYMBOL = '*'
STOP_SYMBOL = 'STOP'
MINUS_INFINITY_SENTENCE_LOG_PROB = -1000

# TODO: IMPLEMENT THIS FUNCTION
# Calculates unigram, bigram, and trigram probabilities given a training corpus
# training_corpus: is a list of the sentences. Each sentence is a string with tokens separated by spaces, ending in a newline character.
# This function outputs three python dictionaries, where the keys are tuples expressing the ngram and the value is the log probability of that ngram
def calc_probabilities(training_corpus):

    from nltk.tokenize import RegexpTokenizer

    unigram_p = {}
    bigram_p = {}
    trigram_p = {}

    unigrams = {}
    bigrams = {}
    trigrams = {}

    # Create unigram, bigram and trigrams
    for sentence in training_corpus:

        # Create tokens
        tokens = sentence.strip().split()

        # Build Unigram Dictionary
        tokens = tokens + [STOP_SYMBOL]
        for word in tokens:
            if word in unigrams:
                unigrams[word] += 1
            else:
                unigrams[word] = 1

        # Build Bigram Dictionary
        tokens = [START_SYMBOL] + tokens
        bigram_tuples = tuple(nltk.bigrams(tokens))
        for bigram in bigram_tuples:
            if bigram in bigrams:
                bigrams[bigram] += 1
            else:
                bigrams[bigram] = 1

        # Build Trigram Dictionary
        tokens = [START_SYMBOL] + tokens
        trigram_tuples = tuple(nltk.trigrams(tokens))
        for trigram in trigram_tuples:
            if trigram in trigrams:
                trigrams[trigram] += 1
            else:
                trigrams[trigram] = 1

    # Calculate Unigram Probabilities
    word_count = sum(unigrams.itervalues())
    unigram_p = {tuple([unigram]): math.log(float(count) / word_count, 2) for unigram, count in unigrams.iteritems()}

    # Calculate Bigram Probabilities
    unigrams[(START_SYMBOL)] = len(training_corpus)
    bigram_p = {bigram: math.log(float(count) / unigrams[bigram[0]], 2) for bigram, count in bigrams.iteritems()}

    # Calculate Trigram Probabilities
    bigrams[(START_SYMBOL, START_SYMBOL)] = len(training_corpus)
    trigram_p = {trigram: math.log(float(count) / bigrams[trigram[:2]], 2) for trigram, count in trigrams.iteritems()}

    return unigram_p, bigram_p, trigram_p

# Prints the output for q1
# Each input is a python dictionary where keys are a tuple expressing the ngram, and the value is the log probability of that ngram
def q1_output(unigrams, bigrams, trigrams, filename):
    # output probabilities
    outfile = open(filename, 'w')

    unigrams_keys = unigrams.keys()
    unigrams_keys.sort()
    for unigram in unigrams_keys:
        outfile.write('UNIGRAM ' + unigram[0] + ' ' + str(unigrams[unigram]) + '\n')

    bigrams_keys = bigrams.keys()
    bigrams_keys.sort()
    for bigram in bigrams_keys:
        outfile.write('BIGRAM ' + bigram[0] + ' ' + bigram[1]  + ' ' + str(bigrams[bigram]) + '\n')

    trigrams_keys = trigrams.keys()
    trigrams_keys.sort()    
    for trigram in trigrams_keys:
        outfile.write('TRIGRAM ' + trigram[0] + ' ' + trigram[1] + ' ' + trigram[2] + ' ' + str(trigrams[trigram]) + '\n')

    outfile.close()


# TODO: IMPLEMENT THIS FUNCTION
# Calculates scores (log probabilities) for every sentence
# ngram_p: python dictionary of probabilities of uni-, bi- and trigrams.
# n: size of the ngram you want to use to compute probabilities
# corpus: list of sentences to score. Each sentence is a string with tokens separated by spaces, ending in a newline character.
# This function must return a python list of scores, where the first element is the score of the first sentence, etc. 
def score(ngram_p, n, corpus):
    scores = []

    # Score sentence with unigram
    if n == 1:
        for sentence in corpus:
            tokens = sentence.strip().split()
            tokens.append(STOP_SYMBOL)
            prob = 0.0
            for token in tokens:
                token = (token,)
                if ngram_p.has_key(token):
                    prob = prob + ngram_p[token]
                else:
                    prob = MINUS_INFINITY_SENTENCE_LOG_PROB
                    break
            scores.append(prob)

    # Score sentence with bigrams
    if n == 2:
        for sentence in corpus:
            prob = 0.0
            tokens = sentence.strip().split()
            tokens.insert(0, START_SYMBOL)
            tokens.append(STOP_SYMBOL)
            bigrams = tuple(nltk.bigrams(tokens))
            for bigram in bigrams:
                if ngram_p.has_key(bigram):
                    prob = prob + ngram_p[bigram]
                else:
                    prob = MINUS_INFINITY_SENTENCE_LOG_PROB
                    break
            scores.append(prob)

    # Score sentence with trigrams
    if n == 3:
        for sentence in corpus:
            prob = 0.0
            tokens = sentence.strip().split()
            tokens.insert(0, START_SYMBOL)
            tokens.insert(0, START_SYMBOL)
            tokens.append(STOP_SYMBOL)
            trigrams = tuple(nltk.trigrams(tokens))
            for trigram in trigrams:
                if ngram_p.has_key(trigram):
                    prob = prob + ngram_p[trigram]
                else:
                    prob = MINUS_INFINITY_SENTENCE_LOG_PROB
                    break
            scores.append(prob)
    return scores

# Outputs a score to a file
# scores: list of scores
# filename: is the output file name
def score_output(scores, filename):
    outfile = open(filename, 'w')
    for score in scores:
        outfile.write(str(score) + '\n')
    outfile.close()

# TODO: IMPLEMENT THIS FUNCTION
# Calculates scores (log probabilities) for every sentence with a linearly interpolated model
# Each ngram argument is a python dictionary where the keys are tuples that express an ngram and the value is the log probability of that ngram
# Like score(), this function returns a python list of scores
def linearscore(unigrams, bigrams, trigrams, corpus):
    scores = []
    return scores

DATA_PATH = 'data/'
OUTPUT_PATH = 'output/'

# DO NOT MODIFY THE MAIN FUNCTION
def main():
    # start timer
    time.clock()

    # get data
    infile = open(DATA_PATH + 'Brown_train.txt', 'r')
    corpus = infile.readlines()
    infile.close()

    # calculate ngram probabilities (question 1)
    unigrams, bigrams, trigrams = calc_probabilities(corpus)

    # question 1 output
    q1_output(unigrams, bigrams, trigrams, OUTPUT_PATH + 'A1.txt')

    # score sentences (question 2)
    uniscores = score(unigrams, 1, corpus)
    biscores = score(bigrams, 2, corpus)
    triscores = score(trigrams, 3, corpus)

    # question 2 output
    score_output(uniscores, OUTPUT_PATH + 'A2.uni.txt')
    score_output(biscores, OUTPUT_PATH + 'A2.bi.txt')
    score_output(triscores, OUTPUT_PATH + 'A2.tri.txt')

    # linear interpolation (question 3)
    linearscores = linearscore(unigrams, bigrams, trigrams, corpus)

    # question 3 output
    score_output(linearscores, OUTPUT_PATH + 'A3.txt')

    # open Sample1 and Sample2 (question 5)
    infile = open(DATA_PATH + 'Sample1.txt', 'r')
    sample1 = infile.readlines()
    infile.close()
    infile = open(DATA_PATH + 'Sample2.txt', 'r')
    sample2 = infile.readlines()
    infile.close() 

    # score the samples
    sample1scores = linearscore(unigrams, bigrams, trigrams, sample1)
    sample2scores = linearscore(unigrams, bigrams, trigrams, sample2)

    # question 5 output
    score_output(sample1scores, OUTPUT_PATH + 'Sample1_scored.txt')
    score_output(sample2scores, OUTPUT_PATH + 'Sample2_scored.txt')

    # print total time to run Part A
    print "Part A time: " + str(time.clock()) + ' sec'

if __name__ == "__main__": main()
