import A
from sklearn.feature_extraction import DictVectorizer
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag, word_tokenize
from nltk.util import ngrams
import math
from collections import defaultdict



# You might change the window size
window_size = 20
local_context_size = 3
collocation_size = 3

def merge_dicts(*dict_args):
    '''
    Given any number of dicts, shallow copy and merge into a new dict,
    precedence goes to key value pairs in latter dicts.
    '''
    result = {}
    for dictionary in dict_args:
        result.update(dictionary)
    return result

def extract_local_context(left, right):
    '''
    Given a left and right context, formatted as a list of items (words or pos), returns a dictionary of including
    the item, it's position from the ambiguous word
    :param left:
    :param right:
    :return feature list:  a list of tuples of the form [(item1, position1), (item2, position2)..]
    '''

    local_context = []
    position = 0

    # Format left context
    start = len(left) - 1
    stop = max(0, start - local_context_size)
    for i in range(start, stop, -1):
        item = left[i]
        position = position - 1
        local_context.append((item, position))
        if position == -(window_size):
            break
    # Format right context
    for i in range(local_context_size):
        item = right[i]
        position = i + 1
        local_context.append((item, position))

    return local_context


# B.1.a,b,c,d
def extract_features(data):
    '''
    :param data: list of instances for a given lexelt with the following structure:
        {
			[(instance_id, left_context, head, right_context, sense_id), ...]
        }
    :return: features: A dictionary with the following structure
             { instance_id: {f1:count, f2:count,...}
            ...
            }
            labels: A dictionary with the following structure
            { instance_id : sense_id }
    '''

    features = {}
    labels = {}

    '''
    The feature sets are described as follows
        F1: Unordered content words in the large context
        F2: A set of words assigned with their positions in the local context
        F3: A set of parts-of-speech tags assigned with their positions in the local context
        F4: A set of bigram, and trigram collocations
        F5: A set of bigram, and trigram collocations of parts-of-speech

    Note: Stopwords, punctuation, and numbers are removed.  Words are lemmatized and lower-case.
    '''

    ns_c = defaultdict(dict)
    n_c = defaultdict(dict)
    for instance in data:

        # Prepare left and right contexts, free of punctuation and lemmatized, with pos tags
        text = instance[1] + instance[2] + instance[3]                      # Combine text to include the head
        tokenizer = RegexpTokenizer(r'\w+')                                 # Remove punctuation
        tokens = tokenizer.tokenize(text)
        head_marker = len(tokenizer.tokenize(instance[1]))
        tokens_pos = pos_tag(tokens)                                        # Create POS tags and lemmatize
        wnl = WordNetLemmatizer()
        tokens_lemma = [wnl.lemmatize(i, j[0].lower()) if j[0].lower() in ['a', 'n', 'v'] else wnl.lemmatize(i)
                        for i, j in tokens_pos]
        left_context_pos = tokens_pos[0:head_marker]                   # Split POS and Lemma into left and right
        right_context_pos = tokens_pos[head_marker + 1:]               # Contexts
        left_context_lemma = tokens_lemma[0:head_marker]
        right_context_lemma = tokens_lemma[head_marker + 1:]

        # Feature Set 1: Unordered Content Words in Large Context
        feature_dict_1 = {}
        context = left_context_lemma + right_context_lemma
        context_set = set(context)
        feature_dict_1 = {word: context.count(word) for word in context_set}

        # Feature Set 2: A set of words assigned with their positions in the local context
        feature_dict_2 = {}
        local_context_words = extract_local_context(left_context_lemma, right_context_lemma)
        feature_dict_2 = {key: 1 for key in local_context_words}

        # Features Set 3: A set of parts-of-speech tags assigned with their positions in the local context
        feature_dict_3 = {}
        left_context_pos_list = [key[1] for key in left_context_pos]
        right_context_pos_list = [key[1] for key in right_context_pos]
        local_context_pos = extract_local_context(left_context_pos_list, right_context_pos_list)
        feature_dict_3 = {key: 1 for key in local_context_pos}

        # Feature Set 4: A set of bigram, and trigram collocations within the collocation window
        feature_dict_4 = {}
        left_marker = head_marker - collocation_size
        right_marker = head_marker + collocation_size
        local_context_head_word = tokens[left_marker:right_marker + 1]
        fourgrams = ngrams(local_context_head_word, 4)
        trigrams = nltk.trigrams(local_context_head_word[1:len(local_context_head_word)-1])
        bigrams = nltk.bigrams(local_context_head_word[2:len(local_context_head_word)-2])
        freq_bigrams = nltk.FreqDist(bigrams)
        freq_trigrams = nltk.FreqDist(trigrams)
        freq_fourgrams = nltk.FreqDist(fourgrams)
        for k, v in freq_bigrams.iteritems():
            feature_dict_4[k] = v
        for k, v in freq_trigrams.iteritems():
            feature_dict_4[k] = v
        for k, v in freq_fourgrams.iteritems():
            feature_dict_4[k] = v

        # Feature Set 5: A set of bigram, and trigram collocations of parts-of-speech
        feature_dict_5 = {}
        left_context_pos_list = left_context_pos_list[left_marker:head_marker]
        right_context_pos_list = right_context_pos_list[0:collocation_size]
        left_context_pos_list.append(instance[2])
        local_context_head_pos = left_context_pos_list + right_context_pos_list
        fourgrams = ngrams(local_context_head_pos , 4)
        trigrams = nltk.trigrams(local_context_head_pos[1:len(local_context_head_pos)-1])
        bigrams = nltk.bigrams(local_context_head_pos[2:len(local_context_head_pos)-2])
        freq_bigrams = nltk.FreqDist(bigrams)
        freq_trigrams = nltk.FreqDist(trigrams)
        freq_fourgrams = nltk.FreqDist(fourgrams)
        for k, v in freq_bigrams.iteritems():
            feature_dict_5[k] = v
        for k, v in freq_trigrams.iteritems():
            feature_dict_5[k] = v
        for k, v in freq_fourgrams.iteritems():
            feature_dict_5[k] = v

        # Feature Set 6: Create data structure used to calculate relevance scores for feature 6
        if instance[4] != '':
            large_context = left_context_lemma[-window_size:] + right_context_lemma[:window_size]
            large_context_nonstop = [word for word in large_context if word not in stopwords.words('english')]
            large_context_nonstop_set = set(large_context_nonstop)
            for c in large_context_nonstop_set:
                if c in n_c:
                    n_c[c] += 1
                else:
                    n_c[c] = 1
                key = (instance[4], c)
                if key in ns_c:
                    ns_c[key] += 1
                else:
                    ns_c[key] = 1

        labels[instance[0]] = instance[4]

        features[instance[0]] = merge_dicts(feature_dict_1, feature_dict_2, feature_dict_3, feature_dict_4, feature_dict_5)

    # Feature Set 6: Calculate relevance scores
    relevance = []
    for k in ns_c:
        s = k[0]
        c = k[1]
        print 'key is', k, 'sense is', s, 'word is', c, 'ns_c is', ns_c[k], 'n_c is', n_c[c]
        prob_s_c = ns_c[k] / float(n_c[c])
        prob_s_not_c = 1 - prob_s_c
        rel = math.log(prob_s_c / float(prob_s_not_c, 2)
        relevance.append([s, c, rel])

    print relevance

    return features, labels

# implemented for you
def vectorize(train_features,test_features):
    '''
    convert set of features to vector representation
    :param train_features: A dictionary with the following structure
             { instance_id: {f1:count, f2:count,...}
            ...
            }
    :param test_features: A dictionary with the following structure
             { instance_id: {f1:count, f2:count,...}
            ...
            }
    :return: X_train: A dictionary with the following structure
             { instance_id: [f1_count,f2_count, ...]}
            ...
            }
            X_test: A dictionary with the following structure
             { instance_id: [f1_count,f2_count, ...]}
            ...
            }
    '''
    X_train = {}
    X_test = {}

    vec = DictVectorizer()
    vec.fit(train_features.values())
    for instance_id in train_features:
        X_train[instance_id] = vec.transform(train_features[instance_id]).toarray()[0]

    for instance_id in test_features:
        X_test[instance_id] = vec.transform(test_features[instance_id]).toarray()[0]

    return X_train, X_test

#B.1.e
def feature_selection(X_train,X_test,y_train):
    '''
    Try to select best features using good feature selection methods (chi-square or PMI)
    or simply you can return train, test if you want to select all features
    :param X_train: A dictionary with the following structure
             { instance_id: [f1_count,f2_count, ...]}
            ...
            }
    :param X_test: A dictionary with the following structure
             { instance_id: [f1_count,f2_count, ...]}
            ...
            }
    :param y_train: A dictionary with the following structure
            { instance_id : sense_id }
    :return:
    '''



    # implement your code here

    #return X_train_new, X_test_new
    # or return all feature (no feature selection):
    return X_train, X_test

# B.2
def classify(X_train, X_test, y_train):
    '''
    Train the best classifier on (X_train, and y_train) then predict X_test labels

    :param X_train: A dictionary with the following structure
            { instance_id: [w_1 count, w_2 count, ...],
            ...
            }

    :param X_test: A dictionary with the following structure
            { instance_id: [w_1 count, w_2 count, ...],
            ...
            }

    :param y_train: A dictionary with the following structure
            { instance_id : sense_id }

    :return: results: a list of tuples (instance_id, label) where labels are predicted by the best classifier
    '''

    results = []


    # implement your code here

    return results

# run part B
def run(train, test, language, answer):
    results = {}

    for lexelt in train:

        train_features, y_train = extract_features(train[lexelt])
        test_features, _ = extract_features(test[lexelt])

        X_train, X_test = vectorize(train_features,test_features)
        X_train_new, X_test_new = feature_selection(X_train, X_test,y_train)
        results[lexelt] = classify(X_train_new, X_test_new,y_train)
        break

    A.print_results(results, answer)