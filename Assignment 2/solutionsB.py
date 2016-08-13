import sys
import nltk
import math
import time
import itertools

START_SYMBOL = '*'
STOP_SYMBOL = 'STOP'
RARE_SYMBOL = '_RARE_'
RARE_WORD_MAX_FREQ = 5
LOG_PROB_OF_ZERO = -1000


# TODO: IMPLEMENT THIS FUNCTION
# Receives a list of tagged sentences and processes each sentence to generate a list of words and a list of tags.
# Each sentence is a string of space separated "WORD/TAG" tokens, with a newline character in the end.
# Remember to include start and stop symbols in yout returned lists, as defined by the constants START_SYMBOL and STOP_SYMBOL.
# brown_words (the list of words) should be a list where every element is a list of the tags of a particular sentence.
# brown_tags (the list of tags) should be a list where every element is a list of the tags of a particular sentence.
def split_wordtags(brown_train):
    brown_words = []
    brown_tags = []

    for sentence in brown_train:
        word_list = []
        tag_list = []
        word_list.append(START_SYMBOL)
        word_list.append(START_SYMBOL)
        tag_list.append(START_SYMBOL)
        tag_list.append(START_SYMBOL)
        tokens = sentence.strip().split()
        for token in tokens:
            i = token.rfind('/')
            word_list.append(token[:i])
            tag_list.append(token[i+1:])
        word_list.append(STOP_SYMBOL)
        tag_list.append(STOP_SYMBOL)
        brown_words.append(word_list)
        brown_tags.append(tag_list)

    return brown_words, brown_tags


# TODO: IMPLEMENT THIS FUNCTION
# This function takes tags from the training data and calculates tag trigram probabilities.
# It returns a python dictionary where the keys are tuples that represent the tag trigram, and the values are the log probability of that trigram
def calc_trigrams(brown_tags):
    q_values = {}
    bigrams = {}
    trigrams = {}

    for sequence in brown_tags:

        # Build Bigram Dictionary
        bigram_tuples = tuple(nltk.bigrams(sequence))
        for bigram in bigram_tuples:
            if bigram in bigrams:
                bigrams[bigram] += 1
            else:
                bigrams[bigram] = 1

        # Build Trigram Dictionary
        trigram_tuples = tuple(nltk.trigrams(sequence))
        for trigram in trigram_tuples:
            if trigram in trigrams:
                trigrams[trigram] += 1
            else:
                trigrams[trigram] = 1

        # Calculate Trigram Probabilities
        q_values = {trigram: math.log(float(count) / bigrams[trigram[:2]], 2) for trigram, count in trigrams.iteritems()}

    return q_values

# This function takes output from calc_trigrams() and outputs it in the proper format
def q2_output(q_values, filename):
    outfile = open(filename, "w")
    trigrams = q_values.keys()
    trigrams.sort()  
    for trigram in trigrams:
        output = " ".join(['TRIGRAM', trigram[0], trigram[1], trigram[2], str(q_values[trigram])])
        outfile.write(output + '\n')
    outfile.close()


# TODO: IMPLEMENT THIS FUNCTION
# Takes the words from the training data and returns a set of all of the words that occur more than 5 times (use RARE_WORD_MAX_FREQ)
# brown_words is a python list where every element is a python list of the words of a particular sentence.
# Note: words that appear exactly 5 times should be considered rare!
def calc_known(brown_words):
    known_words = set([])
    known_words_dict = {}
    known_words_list = []

    # Create known words dictionary with key = word and value = count
    for sentence in brown_words:
        for word in sentence:
            if word in known_words_dict:
                known_words_dict[word] += 1
            else:
                known_words_dict[word] = 1

    # Select words from dictionary where count is > RARE_WORD_MAX_FREQ
    known_words_list = [word for word, count in known_words_dict.iteritems() if count > RARE_WORD_MAX_FREQ]
    known_words = set(known_words_list)

    return known_words

# TODO: IMPLEMENT THIS FUNCTION
# Takes the words from the training data and a set of words that should not be replaced for '_RARE_'
# Returns the equivalent to brown_words but replacing the unknown words by '_RARE_' (use RARE_SYMBOL constant)
def replace_rare(brown_words, known_words):
    brown_words_rare = []

    for sentence in brown_words:
        brown_words_rare.append([RARE_SYMBOL if word not in known_words else word for word in sentence])

    return brown_words_rare

# This function takes the output from replace_rare and outputs it to a file
def q3_output(rare, filename):
    outfile = open(filename, 'w')
    for sentence in rare:
        outfile.write(' '.join(sentence[2:-1]) + '\n')
    outfile.close()


# TODO: IMPLEMENT THIS FUNCTION
# Calculates emission probabilities and creates a set of all possible tags
# The first return value is a python dictionary where each key is a tuple in which the first element is a word
# and the second is a tag, and the value is the log probability of the emission of the word given the tag
# The second return value is a set of all possible tags for this data set
def calc_emission(brown_words_rare, brown_tags):
    e_values = {}
    taglist = set([])
    c_w_t = {}
    c_t = {}

    for word_sent, tag_sent in zip(brown_words_rare, brown_tags):
        for w, t in zip(word_sent, tag_sent):
            w_t = (w, t)
            if w_t in c_w_t:
                c_w_t[w_t] += 1
            else:
                c_w_t[w_t] = 1
            if t in c_t:
                c_t[t] += 1
            else:
                c_t[t] = 1
            if t not in taglist:
                taglist.add(t)

    # Calculate probabilities
    e_values = {w_t: math.log(float(count) / c_t[w_t[1]], 2) for w_t, count in c_w_t.iteritems()}

    return e_values, taglist

# This function takes the output from calc_emissions() and outputs it
def q4_output(e_values, filename):
    outfile = open(filename, "w")
    emissions = e_values.keys()
    emissions.sort()  
    for item in emissions:
        output = " ".join([item[0], item[1], str(e_values[item])])
        outfile.write(output + '\n')
    outfile.close()


# TODO: IMPLEMENT THIS FUNCTION
# This function takes data to tag (brown_dev_words), a set of all possible tags (taglist), a set of all known words (known_words),
# trigram probabilities (q_values) and emission probabilities (e_values) and outputs a list where every element is a tagged sentence 
# (in the WORD/TAG format, separated by spaces and with a newline in the end, just like our input tagged data)
# brown_dev_words is a python list where every element is a python list of the words of a particular sentence.
# taglist is a set of all possible tags
# known_words is a set of all known words
# q_values is from the return of calc_trigrams()
# e_values is from the return of calc_emissions()
# The return value is a list of tagged sentences in the format "WORD/TAG", separated by spaces. Each sentence is a string with a 
# terminal newline, not a list of tokens. Remember also that the output should not contain the "_RARE_" symbol, but rather the
# original words of the sentence!
# This code is based upon the algorithm as described in Collins, M. (2011). Tagging with Hidden Markov Models. New York: Columbia University.


def viterbi(brown_dev_words, taglist, known_words, q_values, e_values):
    tagged = []

    # Initialize tagspace
    tagspace = {}
    taglist = [tag for tag in taglist if tag not in (STOP_SYMBOL, START_SYMBOL)]
    tagspace[-1] = list(START_SYMBOL)
    tagspace[0] = list(START_SYMBOL)

    pi = {}
    bp = {}
    pi[(0, START_SYMBOL, START_SYMBOL)] = 0.0

    for sentence in brown_dev_words:

        n = len(sentence)

        # Create tagspace
        for k in range(1, n+1):
            tagspace[k] = taglist

        # Create copy of sentence with rare words marked with RARE symbol
        tokens = [w if w in known_words else RARE_SYMBOL for w in sentence]

        # Recursively iterate over columns from 1 to k and each possible tag and calculate the maximum probability
        # associated with the cell, defined by the max(sum(v(t-1)) * transition probability * emissions probability
        for k in range(1, n+1):
            for u, v in itertools.product(tagspace[k-1], tagspace[k]):
                max_prob = -float('Inf')
                max_tag = ''
                for w in tagspace[k-2]:
                    prob = pi.get((k-1,w, u), LOG_PROB_OF_ZERO) + q_values.get((w, u, v), LOG_PROB_OF_ZERO) + \
                           e_values.get((tokens[k-1], v), LOG_PROB_OF_ZERO)
                    if prob > max_prob:
                        max_prob = prob
                        max_tag = w
                pi[k, u, v] = max_prob
                bp[k, u, v] = max_tag

        # Get max probability of the last two tokens ending in STOP
        max_prob = -float('Inf')
        for u, v in itertools.product(tagspace[n-1], tagspace[n]):
            prob = pi.get((n, u, v), LOG_PROB_OF_ZERO) + q_values.get((u, v, STOP_SYMBOL), LOG_PROB_OF_ZERO)
            if prob > max_prob:
                max_prob = prob
                max_v = v
                max_u = u

        # Walk back over the backpointer to create the sequence y1...yk
        y = {}
        y[n] = max_v
        y[n - 1] = max_u

        for k in range((n - 2), 0, -1):
            y[k] = bp[k + 2, y[k + 1], y[k + 2]]

        # Format tagged sentence (with rare words included) and append to list
        i = 0
        tagged_sentence = ""
        for tag in y:
            tagged_sentence = tagged_sentence + sentence[i] + '/' + str(y[tag])
            if i < len(sentence) - 1:
                tagged_sentence = tagged_sentence + ' '
            i += 1
        tagged_sentence = tagged_sentence + '\n'
        tagged.append(tagged_sentence)
        break

    return tagged

# This function takes the output of viterbi() and outputs it to file
def q5_output(tagged, filename):
    outfile = open(filename, 'w')
    for sentence in tagged:
        outfile.write(sentence)
    outfile.close()

# TODO: IMPLEMENT THIS FUNCTION
# This function uses nltk to create the taggers described in question 6
# brown_words and brown_tags is the data to be used in training
# brown_dev_words is the data that should be tagged
# The return value is a list of tagged sentences in the format "WORD/TAG", separated by spaces. Each sentence is a string with a 
# terminal newline, not a list of tokens. 
def nltk_tagger(brown_words, brown_tags, brown_dev_words):
    # Hint: use the following line to format data to what NLTK expects for training
    training = [ zip(brown_words[i],brown_tags[i]) for i in xrange(len(brown_words)) ]

    # IMPLEMENT THE REST OF THE FUNCTION HERE
    tagged = []
    tagged_sent = []
    t0 = nltk.DefaultTagger('NOUN')
    t1 = nltk.BigramTagger(training, backoff= t0)
    t2 = nltk.TrigramTagger(training, backoff= t1)
    for sentence in brown_dev_words:
        format_sent = ''
        tagged_sent = t2.tag(sentence)
        for tuple in tagged_sent:
            format_sent = format_sent + tuple[0] + '/' + tuple[1] + ' '
        format_sent += '\n'
        tagged.append(format_sent)

    return tagged

# This function takes the output of nltk_tagger() and outputs it to file
def q6_output(tagged, filename):
    outfile = open(filename, 'w')
    for sentence in tagged:
        outfile.write(sentence)
    outfile.close()

DATA_PATH = 'data/'
OUTPUT_PATH = 'output/'

def main():
    # start timer
    time.clock()

    # open Brown training data
    infile = open(DATA_PATH + "Brown_tagged_train.txt", "r")
    brown_train = infile.readlines()
    infile.close()

    # split words and tags, and add start and stop symbols (question 1)
    brown_words, brown_tags = split_wordtags(brown_train)


    # calculate tag trigram probabilities (question 2)
    q_values = calc_trigrams(brown_tags)


    # question 2 output
    q2_output(q_values, OUTPUT_PATH + 'B2.txt')

    # calculate list of words with count > 5 (question 3)
    known_words = calc_known(brown_words)

    # get a version of brown_words with rare words replace with '_RARE_' (question 3)
    brown_words_rare = replace_rare(brown_words, known_words)

    # question 3 output
    q3_output(brown_words_rare, OUTPUT_PATH + "B3.txt")

    # calculate emission probabilities (question 4)
    e_values, taglist = calc_emission(brown_words_rare, brown_tags)

    # question 4 output
    q4_output(e_values, OUTPUT_PATH + "B4.txt")

    # delete unneceessary data
    del brown_train
    del brown_words_rare

    # open Brown development data (question 5)
    infile = open(DATA_PATH + "Brown_dev.txt", "r")
    brown_dev = infile.readlines()
    infile.close()

    # format Brown development data here
    brown_dev_words = []
    for sentence in brown_dev:
        brown_dev_words.append(sentence.split(" ")[:-1])

    # do viterbi on brown_dev_words (question 5)
    viterbi_tagged = viterbi(brown_dev_words, taglist, known_words, q_values, e_values)

    # question 5 output
    q5_output(viterbi_tagged, OUTPUT_PATH + 'B5.txt')

    # do nltk tagging here
    nltk_tagged = nltk_tagger(brown_words, brown_tags, brown_dev_words)

    # question 6 output
    q6_output(nltk_tagged, OUTPUT_PATH + 'B6.txt')

    # print total time to run Part B
    print "Part B time: " + str(time.clock()) + ' sec'

if __name__ == "__main__": main()
