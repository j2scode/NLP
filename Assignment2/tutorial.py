import nltk
nltk.download('punkt')
nltk.download('tagsets')
nltk.download('brown')
from nltk.corpus import brown

sentence = "At eight o'clock on Thursday morning on Thursday morning on Thursday morning."
tokens = nltk.word_tokenize(sentence)

bigram_tuples = list(nltk.bigrams(tokens))
trigram_tuples = list(nltk.trigrams(tokens))

count = {item: bigram_tuples.count(item) for item in set(bigram_tuples)}
ngrams = [item for item in set(bigram_tuples) if "on" in item]

default_tagger = nltk.DefaultTagger("NN")
tagged_sentence = default_tagger.tag(tokens)

training = brown.tagged_sents(categories='news')

#Create Unigram, Bigram, Trigram taggers based on training set
unigram_tagger = nltk.UnigramTagger(training)
bigram_tagger = nltk.BigramTagger(training, backoff=default_tagger)
trigram_tagger = nltk.TrigramTagger(training, backoff=bigram_tagger)

