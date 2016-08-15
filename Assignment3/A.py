import main
import nltk
from sklearn import svm
from sklearn import neighbors
from nltk.tokenize import RegexpTokenizer


# don't change the window size
window_size = 10

# A.1
def build_s(data):
    '''
    Compute the context vector for each lexelt
    :param data: dic with the following structure:
        {
			lexelt: [(instance_id, left_context, head, right_context, sense_id), ...],
			...
        }
    :return: dic s with the following structure:
        {
			lexelt: [w1,w2,w3, ...],
			...
        }
    '''
    s = {}

    # implement your code here
    for lexelt in data:
        cv = []
        for instance in data[lexelt]:
            left_context = nltk.word_tokenize(instance[1])
            right_context = nltk.word_tokenize(instance[3])
            cv += left_context[-window_size:] + right_context[0:window_size]
        s[lexelt] = list(set(cv))
    return s


# A.1
def vectorize(data, s):
    '''
    :param data: list of instances for a given lexelt with the following structure:
        {
			[(instance_id, left_context, head, right_context, sense_id), ...]
        }
    :param s: list of words (features) for a given lexelt: [w1,w2,w3, ...]
    :return: vectors: A dictionary with the following structure
            { instance_id: [w_1 count, w_2 count, ...],
            ...
            }
            labels: A dictionary with the following structure
            { instance_id : sense_id }
    '''
    vectors = {}
    labels = {}

    # implement your code here
    for instance in data:
        word_count = []
        left_context = nltk.word_tokenize(instance[1])
        right_context = nltk.word_tokenize(instance[3])
        context = left_context[-window_size:] + right_context[0:window_size]
        for word in s:
            word_count.append(context.count(word))
        vectors[instance[0]] = word_count
        labels[instance[0]] = instance[4]

    return vectors, labels


# A.2
def classify(X_train, X_test, y_train):
    '''
    Train two classifiers on (X_train, and y_train) then predict X_test labels
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
    :return: svm_results: a list of tuples (instance_id, label) where labels are predicted by LinearSVC
             knn_results: a list of tuples (instance_id, label) where labels are predicted by KNeighborsClassifier
    '''

    svm_results = []
    knn_results = []

    svm_clf = svm.LinearSVC()
    knn_clf = neighbors.KNeighborsClassifier()

    # Reformat training and test data into vectors
    X_train_values = X_train.values()
    y_train_values = y_train.values()
    X_test_values = X_test.values()

    # Train models
    svm_clf.fit(X_train_values, y_train_values)
    knn_clf.fit(X_train_values, y_train_values)

    # Predict test data based upon trained models
    svm_predictions = svm_clf.predict(X_test_values)
    knn_predictions = knn_clf.predict(X_test_values)

    # Format results into list of tuples
    i = 0
    for instance in X_test:
        svm_tuple = (instance, svm_predictions[i])
        knn_tuple = (instance, knn_predictions[i])
        svm_results.append(svm_tuple)
        knn_results.append(knn_tuple)
        i += 1

    return svm_results, knn_results

# A.3, A.4 output
def print_results(results ,output_file):
    '''
    :param results: A dictionary with key = lexelt and value = a list of tuples (instance_id, label)
    :param output_file: file to write output
    '''

    # implement your code here
    # don't forget to remove the accent of characters using main.replace_accented(input_str)
    # you should sort results on instance_id before printing
    output = []
    #
    for result in results:
        lexelt_item = main.replace_accented(result)
        for tuple in results[result]:
            output_line = []
            instance_id = main.replace_accented(tuple[0])
            sense_id = tuple[1]
            output_line.append(lexelt_item)
            output_line.append(instance_id)
            output_line.append(sense_id)
            output.append(output_line)

    # Sort Output
    output_sorted = sorted(output, key=lambda line: ([line[0], line[1]]))

    # Format output with spaces between the elements of the list
    output_print = []
    for output in output_sorted:
        output_format = output[0] + ' ' + output[1] + ' ' + output[2]
        output_print.append(output_format)

    # Write results to file
    outfile = open(output_file, 'w')
    for output in output_print:
        try:
            outfile.write(output + '\n')
        except:
            print 'unicode error for', output
    outfile.close()

# run part A
def run(train, test, language, knn_file, svm_file):
    s = build_s(train)
    svm_results = {}
    knn_results = {}
    for lexelt in s:
        X_train, y_train = vectorize(train[lexelt], s[lexelt])
        X_test, _ = vectorize(test[lexelt], s[lexelt])
        svm_results[lexelt], knn_results[lexelt] = classify(X_train, X_test, y_train)

    print_results(svm_results, svm_file)
    print_results(knn_results, knn_file)
