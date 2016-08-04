import nltk
nltk.download('punkt')
nltk.download('tagsets')
nltk.download('brown')
import math

from nltk.corpus import brown

sentence = "At eight o'clock on Thursday morning on Thursday morning on Thursday morning."
tokens = nltk.word_tokenize(sentence)
print type(tokens)

bigram_tuples = list(nltk.bigrams(tokens))
trigram_tuples = list(nltk.trigrams(tokens))

count_bigram = {item: math.log((bigram_tuples.count(item) / float(len(bigram_tuples))), 2) for item in set(bigram_tuples)}
count_trigram = {item: math.log((trigram_tuples.count(item) / float(len(trigram_tuples))), 2) for item in set(trigram_tuples)}

print count_unigram
print count_bigram
print count_trigram

ngrams = [item for item in set(bigram_tuples) if "on" in item]

default_tagger = nltk.DefaultTagger("NN")
tagged_sentence = default_tagger.tag(tokens)

training = brown.tagged_sents(categories='news')

#Create Unigram, Bigram, Trigram taggers based on training set
unigram_tagger = nltk.UnigramTagger(training)
bigram_tagger = nltk.BigramTagger(training, backoff=default_tagger)
trigram_tagger = nltk.TrigramTagger(training, backoff=bigram_tagger)

