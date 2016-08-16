import A
from sklearn.feature_extraction import DictVectorizer
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag



# You might change the window size
window_size = 50
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
        F4: A set of unigram, bigram, and trigram collocations
        F5: A set of unigram, bigram, and trigram collocations of parts-of-speech

    Note: Stopwords, punctuation, and numbers are removed.  Words are lemmatized and lower-case.
    '''

    for instance in data:

        # Feature Set 1: Unordered Content Words in Large Context
        feature_dict_1 = {}
        tokenizer = RegexpTokenizer(r'\w+')
        left_context = tokenizer.tokenize(instance[1])
        right_context = tokenizer.tokenize(instance[3])
        stopwords = nltk.corpus.stopwords.words('english')
        left_context = [w for w in left_context if w not in stopwords]
        right_context = [w for w in right_context if w not in stopwords]
        left_context = left_context[-window_size:]
        right_context = right_context[0:window_size]
        wnl = WordNetLemmatizer()
        left_context = [wnl.lemmatize(i, j[0].lower()) if j[0].lower() in ['a', 'n', 'v'] else wnl.lemmatize(i)
                        for i, j in pos_tag(left_context)]
        right_context = [wnl.lemmatize(i, j[0].lower()) if j[0].lower() in ['a', 'n', 'v'] else wnl.lemmatize(i)
                         for i, j in pos_tag(right_context)]
        context = left_context + right_context
        context_set = set(context)
        feature_dict_1 = {word : context.count(word) for word in context_set}

        # Feature Set 2: A set of words assigned with their positions in the local context
        feature_dict_2 = {}
        feature_list = []
        print 'PRINTING LEFT CONTEXT'
        print left_context
        position = 0
        for i in range(len(left_context)-1, (len(left_context)-1) - local_context_size, -1):
            word = left_context[i]
            position = position - 1
            feature_list.append((word, position))
            if position == -(window_size):
                break
        print '\n', 'PRINTING RIGHT CONTEXT'
        print right_context
        for i in range(local_context_size):
            word = right_context[i]
            position = i + 1
            feature_list.append((word, position))
        feature_dict_2 = {key : feature_list.count(key) for key in feature_list}
        print feature_dict_2

        labels[instance[0]] = instance[4]

        #features = merge_dicts(feature_dict_1, feature_dict_2)

        break

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